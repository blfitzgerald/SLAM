## Last updated 04/03/2025
# Title: L-Moment Calculation
# Author: Benjamin FitzGerald
# Purpose: The goal is to calculate the mean of a watershed's average annual maxima, then calculate the L-moments over that area. 
# Inputs: 1) WSArray.npy, 2) Dur, 3) set number (which job from submit file) 4/5) Annual Maxima files, 6) DateList.npy 7)PointList.np, 8) Precip Data for days in DateList
# Inputs from code are compatible with execudable, LMC.sh, and submit file, LMC.sub though the sys.argv inputs could be replaced by the user. 
# Outputs: LM.$setNum.nc

def Shape2Array(shp, rast, crop):
    xdim=np.size(rast['longitude'].values)
    ydim=np.size(rast['latitude'].values)
    
    # # Now let's get the watershed shapefile into an array
    with fiona.open(shp, "r") as shapefile:
        shapes = [feature["geometry"] for feature in shapefile]

    transform = from_origin(rast.longitude.isel(longitude=1).values,rast.latitude.isel(latitude=-1).values,np.abs(rast.longitude.isel(longitude=2).values-rast.longitude.isel(longitude=1).values) ,np.abs(rast.latitude.isel(latitude=2).values-rast.latitude.isel(latitude=1).values))
    rastertemplate=np.ones((xdim,ydim),dtype='float32')
    


    memfile = MemoryFile()
    rastermask = memfile.open(driver='GTiff',
                              height = rastertemplate.shape[1], width = rastertemplate.shape[0],
                              count=1, dtype=str(rastertemplate.dtype),
                              crs='+proj=longlat +datum=NAD83 +no_defs',
                              transform=transform)

    rastermask.write(rastertemplate,1)
    Array, out_transform = mask(rastermask, shapes, crop=crop, all_touched=True)
    Array=Array[0,:,:]
    
    return Array

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

# Inputs from run files
process=int(sys.argv[1])
dur =sys.argv[2]
latname= sys.argv[3]
lonname=sys.argv[4]
precvar=sys.argv[5]
precpreffix=sys.argv[6]
precsuffix=sys.argv[7]
WSShp=sys.argv[8]+".shp"
group_size=int(sys.argv[9])
DPFile=sys.argv[10]
dpkey=sys.argv[11]

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

OUTPUTKey=sys.argv[17]

print('Just Starting')

# Load Annual Maxima Netcdf and extract info such as diminsion size. 
AM = xr.open_mfdataset("*Maxima.*.nc4")
AM['year'] = AM['year'].astype(int)
latitudes = AM.latitude.values
longitudes = AM.longitude.values
yearlist = AM.year.values
ydim = np.size(latitudes)
xdim=np.size(longitudes)
nYears=np.size(yearlist)
yearf=yearlist[0]
yearl=yearlist[nYears-1]
full_start_array=AM.WAM_start.values.astype("datetime64[ns]")
full_end_array=AM.WAM_end.astype("datetime64[ns]")

amwap_array=AM.WAP.values
indices = np.where(amwap_array > 0)
years=indices[0]
lats=indices[1]
lons=indices[2]

lonlist=np.unique(np.sort(lons))
sorted_groups = [np.sort(lonlist[i:i+group_size]) for i in range(0, len(lonlist), group_size)]
process_lons=sorted_groups[process]

# Find the longitudes used in this process
AMCols=AM.isel(longitude=slice(int(process_lons[0]), int(process_lons[-1]+1)))
amwap_array = AMCols['WAP'].values
start_array=AMCols['WAM_start'].values.astype("datetime64[ns]")
end_array=AMCols['WAM_end'].values.astype("datetime64[ns]")

indices = np.where(np.min(amwap_array, axis=0) > 0)

WSArray = np.flip(Shape2Array(WSShp, AM, True),axis=0)
WSArray[WSArray > 0] = 1
WSArray[WSArray==0]=np.nan
WSShape = np.shape(WSArray)
valid_mask_indices = np.where(WSArray.flatten() == 1)[0]

fullWSArray=Shape2Array(WSShp, AM, False)
maskindices=np.argwhere(np.flip(fullWSArray,axis=0)==1)
WSLonCorner=np.min(maskindices[:,1])
WSLatCorner=np.max(maskindices[:,0])

print('Opening all the files')
AMFilesFull = xr.open_mfdataset(precpreffix+'*'+precsuffix)
print('Opened them')

AMFilesSlice=AMFilesFull.isel(longitude=slice(int(process_lons[0]), int(process_lons[0])+np.shape(WSArray)[1]+np.size(process_lons)))
AMFilesSlice_array=AMFilesSlice[precvar].values
FullFileTimes = AMFilesSlice.time.values  # Extract NumPy array of time

Norm=xr.open_dataset(DPFile)
NormSlice=Norm.isel(longitude=slice(int(process_lons[0]), int(process_lons[0])+np.shape(WSArray)[1]+np.size(process_lons)))
NormSlice_Array=NormSlice.design_rain.values

WSFull= np.flip(Shape2Array(WSShp, AM, False),axis=0)

WSNorm=WSFull*Norm
lat_idx = WSLatCorner
lon_idx = WSLonCorner
lat_start, lat_end = lat_idx-WSShape[0]+1, lat_idx+1
lon_start, lon_end = lon_idx, lon_idx+WSShape[1] 
WS_Norm=WSNorm.isel(latitude=slice(lat_idx-WSShape[0]+1, lat_idx+1), longitude=slice(lon_start,lon_end))
WS_Norm_array=WS_Norm.design_rain.values
WS_Norm_array_flat=WS_Norm_array.flatten()
WNAFVs=WS_Norm_array_flat[valid_mask_indices]

if np.isin(WSLonCorner, process_lons):
    WSAMs=np.full((nYears,int(dur), WSShape[0], WSShape[1]), np.nan)
    wstimes=np.full((nYears,int(dur)),np.nan)
    lat_idx = WSLatCorner
    lon_idx = WSLonCorner

#   Extract region around point
    lat_start, lat_end = lat_idx-WSShape[0]+1, lat_idx+1
    lon_start, lon_end = lon_idx, lon_idx+WSShape[1] 
     
    for yy in np.arange(nYears):
        AMStartHour=full_start_array[yy,lat_idx, lon_idx]
        AMEndHour=full_end_array[yy, lat_idx, lon_idx]
        
        sub_data=AMFilesFull.isel(latitude=slice(lat_idx-WSShape[0]+1, lat_idx+1), longitude=slice(lon_start,lon_end)).sel(time=slice(AMStartHour, AMEndHour))
        sub_data_WS=sub_data*WSArray
        wslats=sub_data.latitude
        wslons=sub_data.longitude
        wstimes[yy,:]=sub_data.time.values
        
        WSAMs[yy,:,:,:]=sub_data_WS.precrate.values
    
    WSAMDS = xr.Dataset(
        {"precrate": (["year", "hour", "latitude", "longitude"], WSAMs),
         "hours": (["year", "hour"], wstimes)
            
        },
        coords={"year":AM.year.values, "hour":np.linspace(1, int(dur), int(dur)), "latitude": wslats, "longitude":wslons},
    )
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
    def binomial(n, k):
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
        n = len(x)
        x = np.sort(x)


        l1 = np.sum(x) / binomial(n, 1)


        # Second L-moment
        comb1 = np.arange(n)
        coefl2 = 0.5 / binomial(n, 2)
        sum_xtrans=0
        for ll in np.arange(n):
            sum_xtrans= sum_xtrans+(( comb1[ll]- comb1[n - ll - 1]) * x[ll])

        l2 = coefl2 * sum_xtrans


        # Third L-moment
        comb3 = [binomial(ll, 2) for ll in np.arange(n)]
        coefl3 = 1.0 / 3.0 / binomial(n, 3)
        sum_xtrans = 0
        for ll in np.arange(n):
            sum_xtrans= sum_xtrans+ (comb3[ll] - 2 * comb1[ll] * comb1[n - ll - 1] + comb3[n - ll - 1]) * x[ll]

        l3 = coefl3 * sum_xtrans / l2



        # Fourth L-moment
        # Fourth L-moment
        comb5 = [binomial(ll, 3) for ll in np.arange(n)]
        coefl4 = 0.25 / binomial(n, 4)
        sum_xtrans = 0
        for ll in np.arange(n):
            sum_xtrans= sum_xtrans+ (comb5[ll] - 3 * comb3[ll] * comb1[n - ll - 1]+ 3 * comb1[ll] * comb3[n - ll - 1]- comb5[n - ll - 1]) * x[ll]
        l4 = coefl4 * sum_xtrans / l2
        return l1, l2/l1, l3, l4 
    
    datashape=np.shape(AMStart)
    nyears=datashape[0]
    nlats=datashape[1]
    nlons=datashape[2]
    
    nmask=len(mask_indices)

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

    results = np.full((nlats,nlons, nOutput*4),np.nan)  # Store results (points, time)
    mask_rows, mask_cols = mask_shape  # Extract mask dimensions

    for i in prange(len(points[0])):  # Parallelize the loop over points
         lat_idx = points[0][i]
         lon_idx = points[1][i]

#         Extract region around point
         lat_start, lat_end = lat_idx-mask_rows+1, lat_idx+1
         lon_start, lon_end = lon_idx, lon_idx+mask_cols 
         
         allAMs=np.zeros((nyears, nmask))

         sub_norm= AllNorms[lat_start:lat_end,lon_start:lon_end]
         sub_norm_flat=sub_norm.flatten()
         sub_norm_flat= sub_norm_flat[mask_indices]

         for yy in np.arange(nyears):
             AMStartHour=AMStart[yy,lat_idx, lon_idx]
             AMEndHour=AMEnd[yy, lat_idx, lon_idx]
             
             time_idx_start=np.where(AMStartHour==FullFileTimes)[0][0]
             time_idx_end=np.where(AMEndHour==FullFileTimes)[0][0]

             sub_data = all_precip_array[time_idx_start:time_idx_end, lat_start:lat_end,lon_start:lon_end]

             sub_data_sum=np.sum(sub_data, 0)
             sub_data_sum_flat=sub_data_sum.flatten()
             
             allAMs[yy,:] = sub_data_sum_flat[mask_indices]
             # Use precomputed mask indices
         
         outCount=0
         if DOMEAN==True:
             meanAM=np.zeros(nmask)
             for mm in np.arange(nmask):
                 ptAMs=allAMs[:,mm]
                 meanAM[mm]=np.mean(ptAMs)
             
             if DOSTANDARD==True:
                 l1, lcv, l3, l4=lm(meanAM)
                 results[lat_idx,lon_idx,(outCount*4)+0]=l1
                 results[lat_idx,lon_idx,(outCount*4)+1]=lcv
                 results[lat_idx,lon_idx,(outCount*4)+2]=l3
                 results[lat_idx,lon_idx,(outCount*4)+3]=l4
                 outCount=outCount+1

             if DONORMALIZED==True:
                 meanAM_normalized=meanAM/sub_norm_flat
                 l1, lcv, l3, l4=lm(meanAM_normalized)
                 results[lat_idx,lon_idx,(outCount*4)+0]=l1
                 results[lat_idx,lon_idx,(outCount*4)+1]=lcv
                 results[lat_idx,lon_idx,(outCount*4)+2]=l3
                 results[lat_idx,lon_idx,(outCount*4)+3]=l4
                 outCount=outCount+1

             if DONORMALIZED==True:
                 meanAM_rescaled=meanAM*WSNorm/sub_norm_flat
                 l1, lcv, l3, l4=lm(meanAM_rescaled)
                 results[lat_idx,lon_idx,(outCount*4)+0]=l1
                 results[lat_idx,lon_idx,(outCount*4)+1]=lcv
                 results[lat_idx,lon_idx,(outCount*4)+2]=l3
                 results[lat_idx,lon_idx,(outCount*4)+3]=l4
                 outCount=outCount+1

         if DOMEDIAN==True:
             medianAM=np.zeros(nmask)
             for mm in np.arange(nmask):
                 ptAMs=allAMs[:,mm]
                 medianAM[mm]=np.median(ptAMs)
             
             if DOSTANDARD==True:
                 l1, lcv, l3, l4=lm(medianAM)
                 results[lat_idx,lon_idx,(outCount*4)+0]=l1
                 results[lat_idx,lon_idx,(outCount*4)+1]=lcv
                 results[lat_idx,lon_idx,(outCount*4)+2]=l3
                 results[lat_idx,lon_idx,(outCount*4)+3]=l4
                 outCount=outCount+1

             if DONORMALIZED==True:
                 medianAM_normalized=medianAM/sub_norm_flat
                 l1, lcv, l3, l4=lm(medianAM_normalized)
                 results[lat_idx,lon_idx,(outCount*4)+0]=l1
                 results[lat_idx,lon_idx,(outCount*4)+1]=lcv
                 results[lat_idx,lon_idx,(outCount*4)+2]=l3
                 results[lat_idx,lon_idx,(outCount*4)+3]=l4
                 outCount=outCount+1

             if DONORMALIZED==True:
                 medianAM_rescaled=medianAM*WSNorm/sub_norm_flat
                 l1, lcv, l3, l4=lm(medianAM_rescaled)
                 results[lat_idx,lon_idx,(outCount*4)+0]=l1
                 results[lat_idx,lon_idx,(outCount*4)+1]=lcv
                 results[lat_idx,lon_idx,(outCount*4)+2]=l3
                 results[lat_idx,lon_idx,(outCount*4)+3]=l4
                 outCount=outCount+1
  
    return results
print('Starting the function')
allLMs=fast_masked_LMs(AMFilesSlice_array, start_array, end_array, valid_mask_indices, WSShape, indices, FullFileTimes,int(dur),WNAFVs,NormSlice_Array, doMean, doMedian, doStandard, doNormalized, doRescaled)
print('Ending the function')
outputCounter=0

if doMean==True:
    if doStandard==True:
        ds = xr.Dataset({
            "l1": (["latitude", "longitude"], allLMs[:,:,(outputCounter*4)+0]),
            "lcv": (["latitude", "longitude"], allLMs[:,:,(outputCounter*4)+1]),
            "l3": (["latitude", "longitude"], allLMs[:,:,(outputCounter*4)+2]),
            "l4": (["latitude", "longitude"], allLMs[:,:,(outputCounter*4)+3])},
            coords={"latitude": latitudes, "longitude":longitudes[process_lons]})

        ds.to_netcdf("LMCol."+str(process)+"."+OUTPUTKey+".base.mean.nc") 
        
        outputCounter=outputCounter+1

    if doNormalized==True:
        ds = xr.Dataset({
            "l1": (["latitude", "longitude"], allLMs[:,:,(outputCounter*4)+0]),
            "lcv": (["latitude", "longitude"], allLMs[:,:,(outputCounter*4)+1]),
            "l3": (["latitude", "longitude"], allLMs[:,:,(outputCounter*4)+2]),
            "l4": (["latitude", "longitude"], allLMs[:,:,(outputCounter*4)+3])},
            coords={"latitude": latitudes, "longitude":longitudes[process_lons]})

        ds.to_netcdf("LMCol."+str(process)+"."+OUTPUTKey+"."+dpkey+".norm.mean.nc")    

        outputCounter=outputCounter+1

    if doRescaled==True:
        ds = xr.Dataset({
            "l1": (["latitude", "longitude"], allLMs[:,:,(outputCounter*4)+0]),
            "lcv": (["latitude", "longitude"], allLMs[:,:,(outputCounter*4)+1]),
            "l3": (["latitude", "longitude"], allLMs[:,:,(outputCounter*4)+2]),
            "l4": (["latitude", "longitude"], allLMs[:,:,(outputCounter*4)+3])},
            coords={"latitude": latitudes, "longitude":longitudes[process_lons]})

        ds.to_netcdf("LMCol."+str(process)+"."+OUTPUTKey+"."+dpkey+".rs.mean.nc")    

        outputCounter=outputCounter+1

if doMedian==True:
    if doStandard==True:
        ds = xr.Dataset({
            "l1": (["latitude", "longitude"], allLMs[:,:,(outputCounter*4)+0]),
            "lcv": (["latitude", "longitude"], allLMs[:,:,(outputCounter*4)+1]),
            "l3": (["latitude", "longitude"], allLMs[:,:,(outputCounter*4)+2]),
            "l4": (["latitude", "longitude"], allLMs[:,:,(outputCounter*4)+3])},
            coords={"latitude": latitudes, "longitude":longitudes[process_lons]})

        ds.to_netcdf("LMCol."+str(process)+"."+OUTPUTKey+".base.median.nc") 
        
        outputCounter=outputCounter+1

    if doNormalized==True:
        ds = xr.Dataset({
            "l1": (["latitude", "longitude"], allLMs[:,:,(outputCounter*4)+0]),
            "lcv": (["latitude", "longitude"], allLMs[:,:,(outputCounter*4)+1]),
            "l3": (["latitude", "longitude"], allLMs[:,:,(outputCounter*4)+2]),
            "l4": (["latitude", "longitude"], allLMs[:,:,(outputCounter*4)+3])},
            coords={"latitude": latitudes, "longitude":longitudes[process_lons]})

        ds.to_netcdf("LMCol."+str(process)+"."+OUTPUTKey+"."+dpkey+".norm.median.nc")    

        outputCounter=outputCounter+1

    if doRescaled==True:
        ds = xr.Dataset({
            "l1": (["latitude", "longitude"], allLMs[:,:,(outputCounter*4)+0]),
            "lcv": (["latitude", "longitude"], allLMs[:,:,(outputCounter*4)+1]),
            "l3": (["latitude", "longitude"], allLMs[:,:,(outputCounter*4)+2]),
            "l4": (["latitude", "longitude"], allLMs[:,:,(outputCounter*4)+3])},
            coords={"latitude": latitudes, "longitude":longitudes[process_lons]})

        ds.to_netcdf("LMCol."+str(process)+"."+OUTPUTKey+"."+dpkey+".rs.median.nc")    

        outputCounter=outputCounter+1





   

    
