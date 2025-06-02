#!/usr/bin/env python3
"""
Hypothesis Test for Comparing L-Moments of Watershed Average Precipitation Data and L-Moments of Watershed and Final Domain Drawing

This script performs statistical analysis of precipitation data using Monte Carlo analysis of Watershed Annual Maxima Record to compare similarity of the L-moments of the transposed watershed shapes.  
It uses False Discovery Rate (FDR) correction to determine which areas pass statistical tests for multiple L-moment statistics. The finds the passing locations contiguous to the untransposed watershed and draws the final domain. 

Author: Benjamin FitzGerald
Created: [Dat
Last Modified: [Date]

Usage:
    python script.py <significance_level> <watershed_shapefile> <rain_variable> <output_key> <composite_method>

Arguments:
    significance_level: Float between 0 and 1 (e.g., 0.05 for 5% significance)
    watershed_shapefile: Path to shapefile without .shp extension
    rain_variable: Name of the precipitation variable in the dataset
    output_key: String identifier for output files
    composite_method: 'mean' or 'median' for temporal aggregation

Outputs:
    - TestResult.[output_key].[significance_level].nc: NetCDF file with test results
    - Domain.[output_key].[significance_level].shp: Shapefile of significant regions
    - Various .npy files with bootstrap test statistics

Dependencies:
    - xarray, numpy, scipy, statsmodels
    - fiona, rasterio, geopandas, shapely
    - lmoments3 for L-moment calculations
"""

import xarray as xr
import numpy as np
import lmoments3 as lm3
import statsmodels.stats.multitest
from scipy.stats import percentileofscore
import sys
import fiona
from rasterio.io import MemoryFile
from rasterio.transform import from_origin
from rasterio.mask import mask
import geopandas as gpd
from shapely.geometry import shape, Polygon, MultiPolygon
from shapely.ops import unary_union
from affine import Affine
from rasterio import features


def remove_isolated_ones_from_index(arr, start_i, start_j):
    """
    Remove isolated 1s from a 2D array using flood-fill algorithm.
    
    This function identifies connected components of 1s starting from a seed point
    and removes any isolated 1s that are not connected to the main component.
    
    Args:
        arr (numpy.ndarray): 2D array to process
        start_i (int): Starting row index for flood-fill
        start_j (int): Starting column index for flood-fill
        
    Returns:
        numpy.ndarray: Modified array with isolated 1s removed
    """
    def is_valid(i, j):
        """Check if coordinates are valid and contain a 1"""
        return 0 <= i < len(arr) and 0 <= j < len(arr[0]) and arr[i][j] == 1

    # Use stack-based flood-fill to mark connected component
    stack = [(start_i, start_j)]

    while stack:
        i, j = stack.pop()
        arr[i][j] = -1  # Mark as visited (part of main component)

        # Check all 4-connected neighbors
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            ni, nj = i + dx, j + dy
            if is_valid(ni, nj):
                stack.append((ni, nj))

    # Remove isolated 1s (any remaining 1s are not connected to main component)
    for i in range(len(arr)):
        for j in range(len(arr[0])):
            if arr[i][j] == 1:
                arr[i][j] = 0
    return arr


def Shape2Array(shp, rast, crop):
    """
    Convert shapefile geometry to raster array using template raster.
    
    Args:
        shp (str): Path to shapefile
        rast (xarray.Dataset): Template raster for coordinate system
        crop (bool): Whether to crop the output to shape bounds
        
    Returns:
        numpy.ndarray: Rasterized shapefile as binary array
    """
    # Read shapefile geometries
    with fiona.open(shp, "r") as shapefile:
        shapes = [feature["geometry"] for feature in shapefile]

    # Create transform from raster coordinates
    transform = from_origin(
        rast.longitude.isel(longitude=1).values,
        rast.latitude.isel(latitude=-1).values,
        np.abs(rast.longitude.isel(longitude=2).values - rast.longitude.isel(longitude=1).values),
        np.abs(rast.latitude.isel(latitude=2).values - rast.latitude.isel(latitude=1).values)
    )
    
    # Create template raster
    rastertemplate = np.ones((xdim, ydim), dtype='float32')

    # Create in-memory raster file
    memfile = MemoryFile()
    rastermask = memfile.open(
        driver='GTiff',
        height=rastertemplate.shape[1], 
        width=rastertemplate.shape[0],
        count=1, 
        dtype=str(rastertemplate.dtype),
        crs='+proj=longlat +datum=NAD83 +no_defs',
        transform=transform
    )

    rastermask.write(rastertemplate, 1)
    
    # Mask raster with shapefile
    Array, out_transform = mask(rastermask, shapes, crop=crop, all_touched=True)
    Array = Array[0, :, :]
    
    return Array


# =============================================================================
# MAIN SCRIPT EXECUTION
# =============================================================================

# Parse command line arguments
gp = float(sys.argv[1])                    # Significance level (e.g., 0.05)
WSShp = sys.argv[2] + '.shp'              # Watershed shapefile path
rainVar = sys.argv[3]                      # Rain variable name
OUTPUTKEY = sys.argv[4]                    # Output identifier
compmethod = sys.argv[5]                   # Composite method ('mean' or 'median')

print(f"Starting L-moments analysis with parameters:")
print(f"  Significance level: {gp}")
print(f"  Watershed shapefile: {WSShp}")
print(f"  Rain variable: {rainVar}")
print(f"  Composite method: {compmethod}")

# =============================================================================
# DATA LOADING AND PREPARATION
# =============================================================================

# Load L-moment grid values and annual maxima for watershed
print("Loading datasets...")
LM = xr.open_mfdataset("LMs.*.nc")                    # L-moment statistics grid
WSAnnualsDS = xr.open_mfdataset("WSAM.*.nc")          # Watershed annual maxima
WSAnnuals = WSAnnualsDS[rainVar].sum(dim='hour', skipna=False).values

# Bootstrap parameters
ntrials = 10000
WSShape = np.shape(WSAnnuals)
print(f"Watershed data shape: {WSShape}")

# =============================================================================
# BOOTSTRAP SAMPLING FOR NULL DISTRIBUTION
# =============================================================================

print(f"Running bootstrap sampling with {ntrials} trials...")

# Generate random year indices for bootstrap sampling
YearList = np.random.randint(0, WSShape[0]-1, size=[WSShape[0], ntrials])
lmoms = np.zeros([5, ntrials])

# Bootstrap loop: randomly sample years, compute composite, calculate L-moments
for tt in range(ntrials):
    if tt % 1000 == 0:
        print(f"  Bootstrap trial {tt}/{ntrials}")
        
    yearsIndices = YearList[:, tt]
    rain = np.zeros((WSShape[1], WSShape[2], np.size(yearsIndices)))
    
    # Extract data for selected years
    for yy in range(np.size(yearsIndices)):
        rain[:, :, yy] = WSAnnuals[yearsIndices[yy], :, :]
    
    # Compute temporal composite (mean or median)
    if compmethod == 'mean':
        composite = np.mean(rain, axis=2)
    elif compmethod == 'median':
        composite = np.median(rain, axis=2)
    
    # Flatten and clean data for L-moment calculation
    flattened_list = composite.flatten()
    filtered_list = flattened_list[~np.isnan(flattened_list)].tolist()
    filtered_list = [element for element in filtered_list if element != 0]
    filtered_list = np.array(filtered_list)
    
    # Calculate L-moment ratios
    lmoms[0:4, tt] = lm3.lmom_ratios(filtered_list, 4)
    lmoms[4, tt] = lmoms[1, tt] / lmoms[0, tt]  # L-CV

# Extract bootstrap distributions for each L-moment
L1Test = lmoms[0, :]      # L1 (mean)
LSkewTest = lmoms[2, :]   # L3 (skewness)
LKurtTest = lmoms[3, :]   # L4 (kurtosis)
LCVTest = lmoms[4, :]     # L-CV (coefficient of variation)

# Save bootstrap test statistics
print("Saving bootstrap test statistics...")
np.save(f'L1Test.{OUTPUTKEY}.{gp}.npy', L1Test)
np.save(f'LSkewTest.{OUTPUTKEY}.{gp}.npy', LSkewTest)
np.save(f'LKurtTest.{OUTPUTKEY}.{gp}.npy', LKurtTest)
np.save(f'LCVTest.{OUTPUTKEY}.{gp}.npy', LCVTest)

# =============================================================================
# PREPARE OBSERVED DATA FOR TESTING
# =============================================================================

print("Preparing observed data for statistical testing...")

# Extract coordinate arrays
latitude = np.array(LM.latitude)
longitude = np.array(LM.longitude)

# Create 2D coordinate grids
TDLatArray = np.zeros([np.size(latitude), np.size(longitude)])
TDLonArray = np.zeros([np.size(latitude), np.size(longitude)])

for aa in range(np.size(latitude)):
    for bb in range(np.size(longitude)):
        TDLatArray[aa, bb] = latitude[aa]
        TDLonArray[aa, bb] = longitude[bb]

# Flatten L-moment arrays and coordinates
L1Flat = LM.l1.values.reshape([np.size(longitude) * np.size(latitude)])
LCVFlat = LM.lcv.values.reshape([np.size(longitude) * np.size(latitude)])
LSkewFlat = LM.l3.values.reshape([np.size(longitude) * np.size(latitude)])
LKurtFlat = LM.l4.values.reshape([np.size(longitude) * np.size(latitude)])

TDLat = TDLatArray.reshape(np.size(longitude) * np.size(latitude))
TDLon = TDLonArray.reshape(np.size(longitude) * np.size(latitude))

# Remove NaN values
non_nan_columns = np.isnan(LCVFlat)

TDLat = TDLat[np.where(non_nan_columns == False)]
TDLon = TDLon[np.where(non_nan_columns == False)]
TDL1Flat = L1Flat[np.where(non_nan_columns == False)]
TDLCVFlat = LCVFlat[np.where(non_nan_columns == False)]
TDLSkewFlat = LSkewFlat[np.where(non_nan_columns == False)]
TDLKurtFlat = LKurtFlat[np.where(non_nan_columns == False)]

# =============================================================================
# STATISTICAL TESTING: CALCULATE P-VALUES
# =============================================================================

print("Calculating p-values for each location...")

# Calculate p-values for each location against bootstrap distributions
testSamples = np.size(TDLon)
L1p = np.zeros(testSamples)
L3p = np.zeros(testSamples)
L4p = np.zeros(testSamples)
LCVp = np.zeros(testSamples)

for pp in range(testSamples):
    if pp % 10000 == 0:
        print(f"  Processing location {pp}/{testSamples}")
        
    L1 = TDL1Flat[pp]
    L3 = TDLSkewFlat[pp]
    L4 = TDLKurtFlat[pp]
    LCV = TDLCVFlat[pp]
    
    # Calculate two-tailed p-values
    L1p[pp] = 0.5 - np.abs((percentileofscore(L1Test, L1) - 50) / 100)
    L3p[pp] = 0.5 - np.abs((percentileofscore(LSkewTest, L3) - 50) / 100)
    L4p[pp] = 0.5 - np.abs((percentileofscore(LKurtTest, L4) - 50) / 100)
    LCVp[pp] = 0.5 - np.abs((percentileofscore(LCVTest, LCV) - 50) / 100)

# Save p-values
Pvals = np.zeros((4, np.size(L1p)))
Pvals[0, :] = L1p
Pvals[1, :] = LCVp
Pvals[2, :] = L3p
Pvals[3, :] = L4p
np.save(f'Pvals.{OUTPUTKEY}.{gp}.npy', Pvals)

# =============================================================================
# FALSE DISCOVERY RATE CORRECTION
# =============================================================================

print("Applying False Discovery Rate correction for multiple testing...")

# FDR correction for L1 (mean)
print("Processing L1...")
maxed = 0
L1p1 = np.copy(L1p)
L1Lon = np.copy(TDLon)
L1Lat = np.copy(TDLat)

# Remove zero p-values
L1Lon = L1Lon[L1p1 > 0]
L1Lat = L1Lat[L1p1 > 0]
L1p1 = L1p1[L1p1 > 0]

# Iteratively remove worst points until all remaining points pass FDR
while maxed == 0:
    domainSize = np.size(L1p1)
    L1Reject, L1Pval = statsmodels.stats.multitest.fdrcorrection(
        L1p1, alpha=gp, method='poscorr', is_sorted=False
    )
    sigSize = domainSize - np.sum(L1Reject)

    if domainSize == sigSize:
        maxed = 1
    else:
        minVal = np.argmin(L1p1)
        L1p1 = np.delete(L1p1, minVal)
        L1Lon = np.delete(L1Lon, minVal)
        L1Lat = np.delete(L1Lat, minVal)

print(f'L1 FDR correction complete - {np.size(L1p1)} locations passed')

# FDR correction for L3 (skewness)
print("Processing L3...")
maxed = 0
L3p3 = np.copy(L3p)
L3Lon = np.copy(TDLon)
L3Lat = np.copy(TDLat)

L3Lon = L3Lon[L3p3 > 0]
L3Lat = L3Lat[L3p3 > 0]
L3p3 = L3p3[L3p3 > 0]

while maxed == 0:
    domainSize = np.size(L3p3)
    L3Reject, L3Pval = statsmodels.stats.multitest.fdrcorrection(
        L3p3, alpha=gp, method='poscorr', is_sorted=False
    )
    sigSize = domainSize - np.sum(L3Reject)

    if domainSize == sigSize:
        maxed = 1
    else:
        minVal = np.argmin(L3p3)
        L3p3 = np.delete(L3p3, minVal)
        L3Lon = np.delete(L3Lon, minVal)
        L3Lat = np.delete(L3Lat, minVal)

print(f'L3 FDR correction complete - {np.size(L3p3)} locations passed')

# FDR correction for L4 (kurtosis)
print("Processing L4...")
maxed = 0
L4Lon = np.copy(TDLon)
L4Lat = np.copy(TDLat)
L4p4 = np.copy(L4p)

L4Lon = L4Lon[L4p4 > 0]
L4Lat = L4Lat[L4p4 > 0]
L4p4 = L4p4[L4p4 > 0]

while maxed == 0:
    domainSize = np.size(L4p4)
    L4Reject, L4Pval = statsmodels.stats.multitest.fdrcorrection(
        L4p4, alpha=gp, method='poscorr', is_sorted=False
    )
    sigSize = domainSize - np.sum(L4Reject)

    if domainSize == sigSize:
        maxed = 1
    else:
        minVal = np.argmin(L4p4)
        L4p4 = np.delete(L4p4, minVal)
        L4Lon = np.delete(L4Lon, minVal)
        L4Lat = np.delete(L4Lat, minVal)

print(f'L4 FDR correction complete - {np.size(L4p4)} locations passed')

# FDR correction for LCV (coefficient of variation)
print("Processing LCV...")
maxed = 0
LCVLon = np.copy(TDLon)
LCVLat = np.copy(TDLat)
LCVpCV = np.copy(LCVp)

LCVLon = LCVLon[LCVpCV > 0]
LCVLat = LCVLat[LCVpCV > 0]
LCVpCV = LCVpCV[LCVpCV > 0]

while maxed == 0:
    domainSize = np.size(LCVpCV)
    LCVReject, LCVPval = statsmodels.stats.multitest.fdrcorrection(
        LCVpCV, alpha=gp, method='poscorr', is_sorted=False
    )
    sigSize = domainSize - np.sum(LCVReject)

    if domainSize == sigSize:
        maxed = 1
    else:
        minVal = np.argmin(LCVpCV)
        LCVpCV = np.delete(LCVpCV, minVal)
        LCVLon = np.delete(LCVLon, minVal)
        LCVLat = np.delete(LCVLat, minVal)

print(f'LCV FDR correction complete - {np.size(LCVpCV)} locations passed')

# =============================================================================
# CREATE RESULTS GRIDS
# =============================================================================

print("Creating results grids...")

# Initialize result arrays
PassFlags = np.zeros([np.size(latitude), np.size(longitude)])
L1Result = np.zeros([np.size(latitude), np.size(longitude)])
LCVResult = np.zeros([np.size(latitude), np.size(longitude)])
L3Result = np.zeros([np.size(latitude), np.size(longitude)])
L4Result = np.zeros([np.size(latitude), np.size(longitude)])

# Mark locations that passed each L-moment test
for aa in range(np.size(L1Lon)):
    xx = np.where(longitude == L1Lon[aa])
    yy = np.where(latitude == L1Lat[aa])
    PassFlags[yy, xx] = PassFlags[yy, xx] + 1
    L1Result[yy, xx] = 1

for aa in range(np.size(LCVLon)):
    xx = np.where(longitude == LCVLon[aa])
    yy = np.where(latitude == LCVLat[aa])
    PassFlags[yy, xx] = PassFlags[yy, xx] + 1
    LCVResult[yy, xx] = 1

for aa in range(np.size(L3Lon)):
    xx = np.where(longitude == L3Lon[aa])
    yy = np.where(latitude == L3Lat[aa])
    PassFlags[yy, xx] = PassFlags[yy, xx] + 1
    L3Result[yy, xx] = 1

for aa in range(np.size(L4Lon)):
    xx = np.where(longitude == L4Lon[aa])
    yy = np.where(latitude == L4Lat[aa])
    PassFlags[yy, xx] = PassFlags[yy, xx] + 1
    L4Result[yy, xx] = 1

# =============================================================================
# SAVE RESULTS TO NETCDF
# =============================================================================

print("Saving results to NetCDF...")

dims = ("latitude", "longitude")
coords = {
    "latitude": latitude,
    "longitude": longitude
}

PassingPts = xr.Dataset(
    data_vars=dict(
        PassFlag=(['latitude', 'longitude'], PassFlags),
        L1=(['latitude', 'longitude'], L1Result),
        LCV=(['latitude', 'longitude'], LCVResult),
        L3=(['latitude', 'longitude'], L3Result),
        L4=(['latitude', 'longitude'], L4Result)
    ),
    coords=coords
)

PassingPts.to_netcdf(f"TestResult.{OUTPUTKEY}.{gp}.nc")
Results = PassingPts
output = f"Domain.{OUTPUTKEY}.{gp}.shp"

# =============================================================================
# GENERATE DOMAIN SHAPEFILE
# =============================================================================

print("Generating domain shapefile...")

# Filter to locations that passed all 4 tests
Results = Results.where(Results.PassFlag == 4)
Results['PassFlag'] = Results.PassFlag / Results.PassFlag

# Extract coordinate information
latitude = np.array(Results.latitude)
longitude = np.array(Results.longitude)
ydim = np.size(latitude)
xdim = np.size(longitude)

# Convert watershed shapefile to array
WSArray = Shape2Array(WSShp, Results, True)
WSShape = np.shape(WSArray)
WSFull = Shape2Array(WSShp, Results, False)

# Set up coordinate system for processing
regionLat = Results.latitude
regionLon = Results.longitude
regionLat = np.flip(regionLat)

dx = regionLon[1] - regionLon[0]
dy = regionLat[1] - regionLat[0]

regionX = np.size(regionLon)
regionY = np.size(regionLat)

# Process pass flags array
passArray = np.flip(Results.PassFlag.values, axis=0)
TestDomain = Results.PassFlag.values

# Remove isolated regions using flood-fill
passArray2 = remove_isolated_ones_from_index(
    passArray, 
    np.min(np.where(WSFull == 1)[0]),
    np.min(np.where(WSFull == 1)[1])
)

# Count overlapping valid regions with watershed
testCount = np.zeros([np.size(regionLat) + WSShape[0], np.size(regionLon)])
passCount = np.zeros([np.size(regionLat), np.size(regionLon)])

print("Computing domain overlap...")
for aa in range(regionY):
    for bb in range(regionX):
        if passArray2[aa, bb] == -1:  # Valid connected region
            passCount[aa:aa+WSShape[0], bb:bb+WSShape[1]] = \
                passCount[aa:aa+WSShape[0], bb:bb+WSShape[1]] + WSArray

# =============================================================================
# CONVERT RESULTS TO SHAPEFILE
# =============================================================================

print("Converting results to shapefile...")

# Create xarray Dataset for the final results
dims = ("latitude", "longitude")
coords = {
    "latitude": np.flip(regionLat, axis=0),
    "longitude": regionLon
}

ds = xr.Dataset(
    data_vars=dict(passed=(['latitude', 'longitude'], np.flip(passCount, axis=0))),
    coords=coords
)

# Extract data array and set up coordinate transform
da = ds['passed']
res_x = (da.longitude[1] - da.longitude[0]).item()
res_y = (da.latitude[1] - da.latitude[0]).item()
transform = Affine.translation(
    da.longitude[0] - res_x / 2, 
    da.latitude[0] - res_y / 2
) * Affine.scale(res_x, res_y)

# Convert to binary mask for processing
data = da.fillna(0).values.astype(np.uint8)
binary_mask = (data > 0).astype(np.uint8)

# Extract shapes from binary mask
shapes = features.shapes(binary_mask, mask=binary_mask, transform=transform)

# Convert to shapely geometries
geoms = []
for geom, val in shapes:
    if val == 1:  # Only areas with value 1 (True in binary mask)
        geoms.append(shape(geom))

if geoms:
    print(f"Found {len(geoms)} individual polygons")
    
    # Step 1: Merge all geometries into one
    merged_geom = unary_union(geoms)
    print("Merged into single geometry")
    
    # Step 2: Apply buffer smoothing to fill gaps and smooth edges
    buffer_distance = max(abs(res_x), abs(res_y)) * 0.8
    print(f"Applying buffer smoothing with distance: {buffer_distance:.6f}")
    smoothed_geom = merged_geom.buffer(buffer_distance).buffer(-buffer_distance)
    
    # Step 3: Fill all holes in the polygon
    print("Filling holes...")
    def fill_polygon_holes(geom):
        """Remove all holes from a polygon"""
        if hasattr(geom, 'exterior'):
            return Polygon(geom.exterior)
        return geom
    
    if isinstance(smoothed_geom, MultiPolygon):
        # Handle MultiPolygon case
        filled_parts = []
        total_holes = 0
        for part in smoothed_geom.geoms:
            if hasattr(part, 'interiors'):
                total_holes += len(part.interiors)
            filled_parts.append(fill_polygon_holes(part))
        final_geom = MultiPolygon(filled_parts) if len(filled_parts) > 1 else filled_parts[0]
    else:
        # Handle single Polygon case
        total_holes = len(smoothed_geom.interiors) if hasattr(smoothed_geom, 'interiors') else 0
        final_geom = fill_polygon_holes(smoothed_geom)
    
    print(f"Filled {total_holes} holes")
    
    # Step 4: Simplify geometry to reduce file size
    simplify_tolerance = min(abs(res_x), abs(res_y)) * 0.25
    if simplify_tolerance > 0:
        print(f"Simplifying geometry with tolerance: {simplify_tolerance:.6f}")
        final_geom = final_geom.simplify(simplify_tolerance, preserve_topology=True)
    
    # Create GeoDataFrame with cleaned geometry
    gdf = gpd.GeoDataFrame({
        'geometry': [final_geom],
        'area': [final_geom.area],
        'original_polygons': [len(geoms)]
    }, crs="EPSG:4326")
    
    print(f"Final geometry area: {final_geom.area:.6f}")
    
else:
    # Handle case where no valid geometries were found
    print("No valid geometries found")
    gdf = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")

# Save final shapefile
gdf.to_file(output)
print(f"Analysis complete! Results saved to:")
print(f"  NetCDF: TestResult.{OUTPUTKEY}.{gp}.nc")
print(f"  Shapefile: {output}")
