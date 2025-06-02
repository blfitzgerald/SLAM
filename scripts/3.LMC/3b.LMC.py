#!/usr/bin/env python3
"""
L-Moment Calculation for Watershed Analysis

Purpose:
Calculate the mean of a watershed's average annual maxima, then compute L-moments 
over that area. 

Inputs:
1. WSArray.npy - Watershed array data
2. Duration parameter
3. Set number (job identifier from submit file)
4. Annual Maxima files
5. DateList.npy - List of dates for analysis
6. PointList.npy - Spatial point coordinates
7. Precipitation data for days in DateList

Command Line Arguments:
sys.argv[1]  - process: Process/job number
sys.argv[2]  - dur: Duration parameter
sys.argv[3]  - latname: Latitude variable name
sys.argv[4]  - lonname: Longitude variable name
sys.argv[5]  - precvar: Precipitation variable name
sys.argv[6]  - precpreffix: Precipitation file prefix
sys.argv[7]  - precsuffix: Precipitation file suffix
sys.argv[8]  - WSShp: Watershed shapefile name (without .shp extension)
sys.argv[9]  - group_size: Size of processing groups
sys.argv[10] - DPFile: Design precipitation file
sys.argv[11] - dpkey: Design precipitation key
sys.argv[12] - doMean: Calculate mean statistics (true/false)
sys.argv[13] - doMedian: Calculate median statistics (true/false)
sys.argv[14] - doStandard: Calculate standard L-moments (true/false)
sys.argv[15] - doNormalized: Calculate normalized L-moments (true/false)
sys.argv[16] - doRescaled: Calculate rescaled L-moments (true/false)
sys.argv[17] - OUTPUTKey: Output file identifier

Outputs:
- LM.$setNum.nc - NetCDF files containing L-moment calculations
- WSAM.*.nc - Watershed Annual Maxima files (Only from the job with the actual watershed in it)
- LMCol.*.nc - L-moment column files for different processing configurations

Dependencies:
- xarray: For NetCDF data handling
- numpy: Numerical computations
- fiona: Shapefile reading
- rasterio: Raster data processing
- numba: Just-in-time compilation for performance

Author: Benjamin FitzGerald
Date: 6/2/2025
Version: 2.1
Compatible with: LMC.sh executable and LMC.sub submit file
Note: Designed for HPC cluster environments with job scheduling systems

"""

def Shape2Array(shp, rast, crop):
    """
    Convert shapefile geometry to raster array format.
    
    This function takes a shapefile and rasterizes it to match the dimensions
    and coordinate system of a reference raster dataset.
    
    Parameters:
    -----------
    shp : str
        Path to the shapefile
    rast : xarray.Dataset
        Reference raster dataset for dimensions and coordinates
    crop : bool
        Whether to crop the output to the shapefile bounds
        
    Returns:
    --------
    numpy.ndarray
        2D array where shapefile geometry is represented as values
    """
    # Get dimensions from reference raster
    xdim=np.size(rast['longitude'].values)
    ydim=np.size(rast['latitude'].values)
    
    # Read shapefile geometries
    with fiona.open(shp, "r") as shapefile:
        shapes = [feature["geometry"] for feature in shapefile]

    # Create coordinate transformation based on raster grid
    transform = from_origin(rast.longitude.isel(longitude=1).values,rast.latitude.isel(latitude=-1).values,np.abs(rast.longitude.isel(longitude=2).values-rast.longitude.isel(longitude=1).values) ,np.abs(rast.latitude.isel(latitude=2).values-rast.latitude.isel(latitude=1).values))
    rastertemplate=np.ones((xdim,ydim),dtype='float32')
    
    # Create in-memory raster file
    memfile = MemoryFile()
    rastermask = memfile.open(driver='GTiff',
                              height = rastertemplate.shape[1], width = rastertemplate.shape[0],
                              count=1, dtype=str(rastertemplate.dtype),
                              crs='+proj=longlat +datum=NAD83 +no_defs',
                              transform=transform)

    # Write template and apply mask
    rastermask.write(rastertemplate,1)
    Array, out_transform = mask(rastermask, shapes, crop=crop, all_touched=True)
    Array=Array[0,:,:]
    
    return Array

# Import required libraries
import xarray as xr
import numpy as np
import glob
import math
from numba import njit, prange
import sys
import fiona
from rasterio.io import MemoryFile
from rasterio.transform import from_origin
from rasterio.mask import mask

# Parse command line arguments
process=int(sys.argv[1])          # Process number for parallel execution
dur =sys.argv[2]                  # Duration parameter
latname= sys.argv[3]              # Latitude variable name
lonname=sys.argv[4]               # Longitude variable name  
precvar=sys.argv[5]               # Precipitation variable name
precpreffix=sys.argv[6]           # Precipitation file prefix
precsuffix=sys.argv[7]            # Precipitation file suffix
WSShp=sys.argv[8]+".shp"          # Watershed shapefile path
group_size=int(sys.argv[9])       # Processing group size
DPFile=sys.argv[10]               # Design precipitation file
dpkey=sys.argv[11]                # Design precipitation key

# Parse boolean flags for different calculation types
doMean=sys.argv[12]
doMean=doMean.lower() in ('true')
doMedian=sys.argv[13]
doMedian=doMedian.lower() in ('true')
doStandard=sys.argv[14]
doStandard=doStandard.lower() in ('true')
doNormalized=sys.argv[15]
doNormalized=doNormalized.lower() in ('true')
doRescaled=sys.argv[16]
doRescaled=doRescaled.lower() in ('true')

OUTPUTKey=sys.argv[17]            # Output file identifier

print('Just Starting')

# Load Annual Maxima NetCDF files and extract metadata
# This combines multiple annual maxima files into a single dataset
AM = xr.open_mfdataset("*Maxima.*.nc4")
AM['year'] = AM['year'].astype(int)  # Ensure years are integers

# Extract coordinate and time information
latitudes = AM.latitude.values
longitudes = AM.longitude.values
yearlist = AM.year.values
ydim = np.size(latitudes)
xdim=np.size(longitudes)
nYears=np.size(yearlist)
yearf=yearlist[0]                 # First year
yearl=yearlist[nYears-1]          # Last year

# Extract timing information for annual maxima periods
full_start_array=AM.WAM_start.values.astype("datetime64[ns]")
full_end_array=AM.WAM_end.astype("datetime64[ns]")

# Find valid data points (where WAP > 0)
amwap_array=AM.WAP.values
indices = np.where(amwap_array > 0)
years=indices[0]
lats=indices[1]
lons=indices[2]

# Divide longitude processing into groups for parallel execution
lonlist=np.unique(np.sort(lons))
sorted_groups = [np.sort(lonlist[i:i+group_size]) for i in range(0, len(lonlist), group_size)]
process_lons=sorted_groups[process]

# Extract data for current process longitude range
AMCols=AM.isel(longitude=slice(int(process_lons[0]), int(process_lons[-1]+1)))
amwap_array = AMCols['WAP'].values
start_array=AMCols['WAM_start'].values.astype("datetime64[ns]")
end_array=AMCols['WAM_end'].values.astype("datetime64[ns]")

# Find indices where all years have valid data
indices = np.where(np.min(amwap_array, axis=0) > 0)

# Process watershed shapefile into array format
WSArray = np.flip(Shape2Array(WSShp, AM, True),axis=0)  # Flip to match coordinate system
WSArray[WSArray > 0] = 1          # Convert to binary mask
WSArray[WSArray==0]=np.nan        # Set background to NaN
WSShape = np.shape(WSArray)       # Store watershed shape

# Create flattened index array for valid watershed pixels
valid_mask_indices = np.where(WSArray.flatten() == 1)[0]

# Get full watershed extent for coordinate calculations
fullWSArray=Shape2Array(WSShp, AM, False)
maskindices=np.argwhere(np.flip(fullWSArray,axis=0)==1)
WSLonCorner=np.min(maskindices[:,1])  # Western-most longitude index
WSLatCorner=np.max(maskindices[:,0])  # Northern-most latitude index

print('Opening all the files')
# Load all precipitation files
AMFilesFull = xr.open_mfdataset(precpreffix+'*'+precsuffix)
print('Opened them')

# Slice precipitation data to current processing region
AMFilesSlice=AMFilesFull.isel(longitude=slice(int(process_lons[0]), int(process_lons[0])+np.shape(WSArray)[1]+np.size(process_lons)))
AMFilesSlice_array=AMFilesSlice[precvar].values
FullFileTimes = AMFilesSlice.time.values  # Extract time array

# Load normalization/design precipitation data
Norm=xr.open_dataset(DPFile)
NormSlice=Norm.isel(longitude=slice(int(process_lons[0]), int(process_lons[0])+np.shape(WSArray)[1]+np.size(process_lons)))
NormSlice_Array=NormSlice.design_rain.values

# Apply watershed mask to full dataset
WSFull= np.flip(Shape2Array(WSShp, AM, False),axis=0)

# Calculate watershed-specific normalization values
WSNorm=WSFull*Norm
lat_idx = WSLatCorner
lon_idx = WSLonCorner
lat_start, lat_end = lat_idx-WSShape[0]+1, lat_idx+1
lon_start, lon_end = lon_idx, lon_idx+WSShape[1] 
WS_Norm=WSNorm.isel(latitude=slice(lat_idx-WSShape[0]+1, lat_idx+1), longitude=slice(lon_start,lon_end))
WS_Norm_array=WS_Norm.design_rain.values
WS_Norm_array_flat=WS_Norm_array.flatten()
WNAFVs=WS_Norm_array_flat[valid_mask_indices]  # Normalization values for valid pixels

# Extract watershed annual maxima if watershed corner is in current process
if np.isin(WSLonCorner, process_lons):
    # Initialize arrays for watershed annual maxima
    WSAMs=np.full((nYears,int(dur), WSShape[0], WSShape[1]), np.nan)
    wstimes=np.full((nYears,int(dur)),np.nan)
    lat_idx = WSLatCorner
    lon_idx = WSLonCorner

    # Define extraction region around watershed
    lat_start, lat_end = lat_idx-WSShape[0]+1, lat_idx+1
    lon_start, lon_end = lon_idx, lon_idx+WSShape[1] 
     
    # Extract annual maxima periods for each year
    for yy in np.arange(nYears):
        AMStartHour=full_start_array[yy,lat_idx, lon_idx]
        AMEndHour=full_end_array[yy, lat_idx, lon_idx]
        
        # Extract precipitation data for current annual maximum period
        sub_data=AMFilesFull.isel(latitude=slice(lat_idx-WSShape[0]+1, lat_idx+1), longitude=slice(lon_start,lon_end)).sel(time=slice(AMStartHour, AMEndHour))
        sub_data_WS=sub_data*WSArray  # Apply watershed mask
        wslats=sub_data.latitude
        wslons=sub_data.longitude
        wstimes[yy,:]=sub_data.time.values
        
        WSAMs[yy,:,:,:]=sub_data_WS.precrate.values
    
    # Create watershed annual maxima dataset
    WSAMDS = xr.Dataset(
        {"precrate": (["year", "hour", "latitude", "longitude"], WSAMs),
         "hours": (["year", "hour"], wstimes)
            
        },
        coords={"year":AM.year.values, "hour":np.linspace(1, int(dur), int(dur)), "latitude": wslats, "longitude":wslons},
    )
    
    # Save different versions based on flags
    if doStandard==True:
        WSAMDS.to_netcdf("WSAM."+OUTPUTKey+".base.nc")

    if doRescaled==True:
        WSAMDS.to_netcdf("WSAM."+OUTPUTKey+"."+dpkey+".rs.nc")

    if doNormalized==True:
        WSAMDS_norm=WSAMDS
        WSAMDS_norm['precrate']=WSAMDS['precrate']/WS_Norm.design_rain
        WSAMDS_norm.to_netcdf("WSAM."+OUTPUTKey+"."+dpkey+".norm.nc")


@njit(parallel=True)
def fast_masked_LMs(all_precip_array, AMStart, AMEnd, mask_indices, mask_shape, points, FullFileTimes, dur, WSNorm, AllNorms, DOMEAN, DOMEDIAN, DOSTANDARD, DONORMALIZED, DORESCALED):
    """
    Calculate L-moments for masked precipitation data using Numba for performance.
    
    This function computes L-moment statistics (L1, L2, L3, L4) for precipitation data
    within watershed boundaries. L-moments are robust statistical measures that describe
    the shape, scale, and location of probability distributions.
    
    L-moments calculated:
    - L1: Location parameter (mean)
    - L2/L1: L-coefficient of variation (scale)
    - L3: L-skewness (asymmetry)
    - L4: L-kurtosis (tail behavior)
    
    Parameters:
    -----------
    all_precip_array : numpy.ndarray
        3D array of precipitation data [time, lat, lon]
    AMStart, AMEnd : numpy.ndarray
        Start and end times for annual maxima periods
    mask_indices : numpy.ndarray
        Indices of valid watershed pixels in flattened array
    mask_shape : tuple
        Shape of the watershed mask
    points : tuple
        Coordinate indices for processing points
    FullFileTimes : numpy.ndarray
        Time coordinate array
    dur : int
        Duration parameter
    WSNorm : numpy.ndarray
        Watershed normalization values
    AllNorms : numpy.ndarray
        Full normalization array
    DOMEAN, DOMEDIAN : bool
        Flags for mean/median calculations
    DOSTANDARD, DONORMALIZED, DORESCALED : bool
        Flags for different L-moment types
        
    Returns:
    --------
    numpy.ndarray
        4D array of L-moment results [lat, lon, statistic_type*4]
    """
    
    def binomial(n, k):
        """
        Calculate binomial coefficient C(n,k) using log-space computation
        to avoid overflow for large numbers.
        """
        if k > n:
            return 0
        if k == 0 or k == n:
            return 1
        k = min(k, n - k)  # Take advantage of symmetry: C(n, k) == C(n, n-k)
        log_binom = 0.0
        for i in range(k):
            log_binom += np.log(n - i) - np.log(i + 1)
        return np.exp(log_binom)

    def lm(x):
        """
        Calculate L-moments for a given data series.
        
        L-moments are linear combinations of order statistics that provide
        robust measures of distribution properties:
        - More robust to outliers than conventional moments
        - Exist for any distribution with finite mean
        - Bounded for shape parameters
        
        Parameters:
        -----------
        x : numpy.ndarray
            Input data series
            
        Returns:
        --------
        tuple
            (L1, L2/L1, L3, L4) - L-moment statistics
        """
        n = len(x)
        x = np.sort(x)  # L-moments require sorted data

        # First L-moment (L1) - equivalent to arithmetic mean
        l1 = np.sum(x) / binomial(n, 1)

        # Second L-moment (L2) - measure of scale/dispersion
        comb1 = np.arange(n)
        coefl2 = 0.5 / binomial(n, 2)
        sum_xtrans=0
        for ll in np.arange(n):
            sum_xtrans= sum_xtrans+(( comb1[ll]- comb1[n - ll - 1]) * x[ll])

        l2 = coefl2 * sum_xtrans

        # Third L-moment (L3) - measure of skewness
        comb3 = [binomial(ll, 2) for ll in np.arange(n)]
        coefl3 = 1.0 / 3.0 / binomial(n, 3)
        sum_xtrans = 0
        for ll in np.arange(n):
            sum_xtrans= sum_xtrans+ (comb3[ll] - 2 * comb1[ll] * comb1[n - ll - 1] + comb3[n - ll - 1]) * x[ll]

        l3 = coefl3 * sum_xtrans / l2

        # Fourth L-moment (L4) - measure of kurtosis
        comb5 = [binomial(ll, 3) for ll in np.arange(n)]
        coefl4 = 0.25 / binomial(n, 4)
        sum_xtrans = 0
        for ll in np.arange(n):
            sum_xtrans= sum_xtrans+ (comb5[ll] - 3 * comb3[ll] * comb1[n - ll - 1]+ 3 * comb1[ll] * comb3[n - ll - 1]- comb5[n - ll - 1]) * x[ll]
        l4 = coefl4 * sum_xtrans / l2
        return l1, l2/l1, l3, l4 
    
    # Initialize dimensions and output arrays
    datashape=np.shape(AMStart)
    nyears=datashape[0]
    nlats=datashape[1]
    nlons=datashape[2]
    
    nmask=len(mask_indices)  # Number of valid watershed pixels

    # Count number of output statistics based on flags
    nOutput=0
    if DOMEAN==True:
        if DOSTANDARD==True:
            nOutput=nOutput+1
        if DONORMALIZED==True:
            nOutput=nOutput+1
        if DORESCALED==True:
            nOutput=nOutput+1
    
    if DOMEDIAN==True:
        if DOSTANDARD==True:
            nOutput=nOutput+1
        if DONORMALIZED==True:
            nOutput=nOutput+1
        if DORESCALED==True:
            nOutput=nOutput+1

    results = np.full((nlats,nlons, nOutput*4),np.nan)  # Store results (4 L-moments per output type)
    mask_rows, mask_cols = mask_shape  # Extract mask dimensions

    # Parallel loop over all spatial points
    for i in prange(len(points[0])):  # Parallelize the loop over points
         lat_idx = points[0][i]
         lon_idx = points[1][i]

         # Define extraction region around current point
         lat_start, lat_end = lat_idx-mask_rows+1, lat_idx+1
         lon_start, lon_end = lon_idx, lon_idx+mask_cols 
         
         # Initialize array for all annual maxima at this location
         allAMs=np.zeros((nyears, nmask))

         # Extract normalization values for current region
         sub_norm= AllNorms[lat_start:lat_end,lon_start:lon_end]
         sub_norm_flat=sub_norm.flatten()
         sub_norm_flat= sub_norm_flat[mask_indices]

         # Process each year's annual maximum period
         for yy in np.arange(nyears):
             AMStartHour=AMStart[yy,lat_idx, lon_idx]
             AMEndHour=AMEnd[yy, lat_idx, lon_idx]
             
             # Find time indices for current annual maximum period
             time_idx_start=np.where(AMStartHour==FullFileTimes)[0][0]
             time_idx_end=np.where(AMEndHour==FullFileTimes)[0][0]

             # Extract precipitation data for current period and region
             sub_data = all_precip_array[time_idx_start:time_idx_end, lat_start:lat_end,lon_start:lon_end]

             # Sum precipitation over time period
             sub_data_sum=np.sum(sub_data, 0)
             sub_data_sum_flat=sub_data_sum.flatten()
             
             # Extract values for watershed pixels only
             allAMs[yy,:] = sub_data_sum_flat[mask_indices]
         
         # Calculate L-moments for different statistical measures
         outCount=0
         
         # Mean-based L-moments
         if DOMEAN==True:
             meanAM=np.zeros(nmask)
             # Calculate mean across years for each watershed pixel
             for mm in np.arange(nmask):
                 ptAMs=allAMs[:,mm]
                 meanAM[mm]=np.mean(ptAMs)
             
             # Standard (non-normalized) L-moments
             if DOSTANDARD==True:
                 l1, lcv, l3, l4=lm(meanAM)
                 results[lat_idx,lon_idx,(outCount*4)+0]=l1
                 results[lat_idx,lon_idx,(outCount*4)+1]=lcv
                 results[lat_idx,lon_idx,(outCount*4)+2]=l3
                 results[lat_idx,lon_idx,(outCount*4)+3]=l4
                 outCount=outCount+1

             # Normalized L-moments (divided by design precipitation)
             if DONORMALIZED==True:
                 meanAM_normalized=meanAM/sub_norm_flat
                 l1, lcv, l3, l4=lm(meanAM_normalized)
                 results[lat_idx,lon_idx,(outCount*4)+0]=l1
                 results[lat_idx,lon_idx,(outCount*4)+1]=lcv
                 results[lat_idx,lon_idx,(outCount*4)+2]=l3
                 results[lat_idx,lon_idx,(outCount*4)+3]=l4
                 outCount=outCount+1

             # Rescaled L-moments (adjusted by watershed normalization)
             if DONORMALIZED==True:  # Note: This appears to be checking DONORMALIZED instead of DORESCALED
                 meanAM_rescaled=meanAM*WSNorm/sub_norm_flat
                 l1, lcv, l3, l4=lm(meanAM_rescaled)
                 results[lat_idx,lon_idx,(outCount*4)+0]=l1
                 results[lat_idx,lon_idx,(outCount*4)+1]=lcv
                 results[lat_idx,lon_idx,(outCount*4)+2]=l3
                 results[lat_idx,lon_idx,(outCount*4)+3]=l4
                 outCount=outCount+1

         # Median-based L-moments
         if DOMEDIAN==True:
             medianAM=np.zeros(nmask)
             # Calculate median across years for each watershed pixel
             for mm in np.arange(nmask):
                 ptAMs=allAMs[:,mm]
                 medianAM[mm]=np.median(ptAMs)
             
             # Standard (non-normalized) L-moments for median
             if DOSTANDARD==True:
                 l1, lcv, l3, l4=lm(medianAM)
                 results[lat_idx,lon_idx,(outCount*4)+0]=l1
                 results[lat_idx,lon_idx,(outCount*4)+1]=lcv
                 results[lat_idx,lon_idx,(outCount*4)+2]=l3
                 results[lat_idx,lon_idx,(outCount*4)+3]=l4
                 outCount=outCount+1

             # Normalized L-moments for median
             if DONORMALIZED==True:
                 medianAM_normalized=medianAM/sub_norm_flat
                 l1, lcv, l3, l4=lm(medianAM_normalized)
                 results[lat_idx,lon_idx,(outCount*4)+0]=l1
                 results[lat_idx,lon_idx,(outCount*4)+1]=lcv
                 results[lat_idx,lon_idx,(outCount*4)+2]=l3
                 results[lat_idx,lon_idx,(outCount*4)+3]=l4
                 outCount=outCount+1

             # Rescaled L-moments for median
             if DONORMALIZED==True:  # Note: This appears to be checking DONORMALIZED instead of DORESCALED
                 medianAM_rescaled=medianAM*WSNorm/sub_norm_flat
                 l1, lcv, l3, l4=lm(medianAM_rescaled)
                 results[lat_idx,lon_idx,(outCount*4)+0]=l1
                 results[lat_idx,lon_idx,(outCount*4)+1]=lcv
                 results[lat_idx,lon_idx,(outCount*4)+2]=l3
                 results[lat_idx,lon_idx,(outCount*4)+3]=l4
                 outCount=outCount+1
  
    return results

# Execute main L-moment calculation
print('Starting the function')
allLMs=fast_masked_LMs(AMFilesSlice_array, start_array, end_array, valid_mask_indices, WSShape, indices, FullFileTimes,int(dur),WNAFVs,NormSlice_Array, doMean, doMedian, doStandard, doNormalized, doRescaled)
print('Ending the function')

# Save results to NetCDF files based on calculation flags
outputCounter=0

# Save mean-based L-moment results
if doMean==True:
    # Standard mean L-moments
    if doStandard==True:
        ds = xr.Dataset({
            "l1": (["latitude", "longitude"], allLMs[:,:,(outputCounter*4)+0]),    # First L-moment (location)
            "lcv": (["latitude", "longitude"], allLMs[:,:,(outputCounter*4)+1]),   # L-coefficient of variation
            "l3": (["latitude", "longitude"], allLMs[:,:,(outputCounter*4)+2]),    # L-skewness
            "l4": (["latitude", "longitude"], allLMs[:,:,(outputCounter*4)+3])},   # L-kurtosis
            coords={"latitude": latitudes, "longitude":longitudes[process_lons]})

        ds.to_netcdf("LMCol."+str(process)+"."+OUTPUTKey+".base.mean.nc") 
        
        outputCounter=outputCounter+1

    # Normalized mean L-moments
    if doNormalized==True:
        ds = xr.Dataset({
            "l1": (["latitude", "longitude"], allLMs[:,:,(outputCounter*4)+0]),
            "lcv": (["latitude", "longitude"], allLMs[:,:,(outputCounter*4)+1]),
            "l3": (["latitude", "longitude"], allLMs[:,:,(outputCounter*4)+2]),
            "l4": (["latitude", "longitude"], allLMs[:,:,(outputCounter*4)+3])},
            coords={"latitude": latitudes, "longitude":longitudes[process_lons]})

        ds.to_netcdf("LMCol."+str(process)+"."+OUTPUTKey+"."+dpkey+".norm.mean.nc")    

        outputCounter=outputCounter+1

    # Rescaled mean L-moments
    if doRescaled==True:
        ds = xr.Dataset({
            "l1": (["latitude", "longitude"], allLMs[:,:,(outputCounter*4)+0]),
            "lcv": (["latitude", "longitude"], allLMs[:,:,(outputCounter*4)+1]),
            "l3": (["latitude", "longitude"], allLMs[:,:,(outputCounter*4)+2]),
            "l4": (["latitude", "longitude"], allLMs[:,:,(outputCounter*4)+3])},
            coords={"latitude": latitudes, "longitude":longitudes[process_lons]})

        ds.to_netcdf("LMCol."+str(process)+"."+OUTPUTKey+"."+dpkey+".rs.mean.nc")    

        outputCounter=outputCounter+1

# Save median-based L-moment results
if doMedian==True:
    # Standard median L-moments
    if doStandard==True:
        ds = xr.Dataset({
            "l1": (["latitude", "longitude"], allLMs[:,:,(outputCounter*4)+0]),
            "lcv": (["latitude", "longitude"], allLMs[:,:,(outputCounter*4)+1]),
            "l3": (["latitude", "longitude"], allLMs[:,:,(outputCounter*4)+2]),
            "l4": (["latitude", "longitude"], allLMs[:,:,(outputCounter*4)+3])},
            coords={"latitude": latitudes, "longitude":longitudes[process_lons]})

        ds.to_netcdf("LMCol."+str(process)+"."+OUTPUTKey+".base.median.nc") 
        
        outputCounter=outputCounter+1

    # Normalized median L-moments
    if doNormalized==True:
        ds = xr.Dataset({
            "l1": (["latitude", "longitude"], allLMs[:,:,(outputCounter*4)+0]),
            "lcv": (["latitude", "longitude"], allLMs[:,:,(outputCounter*4)+1]),
            "l3": (["latitude", "longitude"], allLMs[:,:,(outputCounter*4)+2]),
            "l4": (["latitude", "longitude"], allLMs[:,:,(outputCounter*4)+3])},
            coords={"latitude": latitudes, "longitude":longitudes[process_lons]})

        ds.to_netcdf("LMCol."+str(process)+"."+OUTPUTKey+"."+dpkey+".norm.median.nc")    

        outputCounter=outputCounter+1

    # Rescaled median L-moments
    if doRescaled==True:
        ds = xr.Dataset({
            "l1": (["latitude", "longitude"], allLMs[:,:,(outputCounter*4)+0]),
            "lcv": (["latitude", "longitude"], allLMs[:,:,(outputCounter*4)+1]),
            "l3": (["latitude", "longitude"], allLMs[:,:,(outputCounter*4)+2]),
            "l4": (["latitude", "longitude"], allLMs[:,:,(outputCounter*4)+3])},
            coords={"latitude": latitudes, "longitude":longitudes[process_lons]})

        ds.to_netcdf("LMCol."+str(process)+"."+OUTPUTKey+"."+dpkey+".rs.median.nc")    

        outputCounter=outputCounter+1

# End of script
print('Script completed successfully')

    
