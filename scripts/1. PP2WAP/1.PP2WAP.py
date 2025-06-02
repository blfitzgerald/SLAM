#!/usr/bin/env python3
"""
Point Precipitation to Watershed Average Precipitation (WAP) Processor

This script processes precipitation data from NetCDF files by transposing the watershed
mask to calculate area-averaged precipitation values within watershed boundaries.
The script uses parallel processing with Numba for efficient computation of 
spatially-weighted precipitation averages.

Methodology:
    1. Loads precipitation NetCDF files and watershed shapefiles
    2. Converts watershed shapefile to raster mask
    3. For each grid point, applies the watershed mask cornered at that point
    4. Calculates area-averaged precipitation within the watershed boundary
    5. Outputs daily WAP values as compressed NetCDF4 files

Usage:
    python 1.PP2WAP.py <watershed_name> <precip_var> <lon_dim> <lat_dim> <timesteps_per_day> <output_format>

Arguments:
    watershed_name      : Base name of watershed shapefile (without .shp extension)
    precip_var         : Name of precipitation variable in NetCDF files
    lon_dim            : Name of longitude dimension in input files
    lat_dim            : Name of latitude dimension in input files  
    timesteps_per_day  : Number of time steps per day in input data
    output_format      : Format identifier for output filename

Input Requirements:
    - NetCDF files with precipitation data (*.nc in current directory)
    - Watershed shapefile (.shp, .shx, .dbf files)
    - Files must have consistent coordinate systems (NAD83 lat/lon assumed)

Output:
    - Daily WAP NetCDF4 files: WAP.YYYYMMDD.<format>.nc4
    - Files are compressed using zlib compression level 9

Dependencies:
    - xarray: NetCDF file handling
    - numpy: Numerical operations
    - numba: JIT compilation for performance
    - fiona: Shapefile reading
    - rasterio: Raster processing and masking

Author: Benjamin FitzGerald
Date: 6/2/2025
Version: 2.1
"""

import xarray as xr
import numpy as np
from numba import njit, prange
import fiona
from rasterio.io import MemoryFile
from rasterio.transform import from_origin
from rasterio.mask import mask
import sys

def Shape2Array(shp, rast, crop):
    """
    Convert shapefile to raster array matching the input NetCDF grid.
    
    Parameters:
    -----------
    shp : str
        Path to shapefile
    rast : xarray.Dataset
        Reference raster dataset for coordinate system
    crop : bool
        Whether to crop the output to the shapefile bounds
        
    Returns:
    --------
    numpy.ndarray
        Binary mask array where 1 = inside watershed, 0 = outside
    """
    # Get dimensions of the reference raster
    xdim = np.size(rast['longitude'].values)
    ydim = np.size(rast['latitude'].values)
    
    # Load watershed shapefile geometries
    with fiona.open(shp, "r") as shapefile:
        shapes = [feature["geometry"] for feature in shapefile]

    # Create geospatial transform from raster coordinates
    # Uses the coordinate spacing to define pixel size
    transform = from_origin(
        rast.longitude.isel(longitude=1).values,
        rast.latitude.isel(latitude=-1).values,
        np.abs(rast.longitude.isel(longitude=2).values - rast.longitude.isel(longitude=1).values),
        np.abs(rast.latitude.isel(latitude=2).values - rast.latitude.isel(latitude=1).values)
    )
    
    # Create template raster filled with ones
    rastertemplate = np.ones((xdim, ydim), dtype='float32')
    
    # Create in-memory raster file for masking operation
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

    # Write template to memory file and apply watershed mask
    rastermask.write(rastertemplate, 1)
    Array, out_transform = mask(rastermask, shapes, crop=crop, all_touched=True)
    Array = Array[0, :, :]
    
    return Array

# Parse command line arguments
WSShp = sys.argv[1] + ".shp"              # Watershed Shapefile path
precvar = sys.argv[2]                     # Precipitation variable name
lonName = sys.argv[3]                     # Longitude dimension name
latName = sys.argv[4]                     # Latitude dimension name
tpd = int(sys.argv[5])                    # Time steps per day
OutputFormat = sys.argv[6]                # Output format identifier

# Load all precipitation NetCDF files in current directory (bc this is to be used in a HTCondor job it will just load all the nc file submitted with the job. 
RAINFILE = xr.open_mfdataset('*.nc').load()

# Standardize coordinate names to 'longitude' and 'latitude'
if 'longitude' not in RAINFILE.dims:
    RAINFILE = RAINFILE.rename({lonName: 'longitude'})
if 'latitude' not in RAINFILE.dims:
    RAINFILE = RAINFILE.rename({latName: 'latitude'})

# Extract key variables and coordinates
data_var = RAINFILE[precvar]              # Select precipitation variable
latitudes = RAINFILE["latitude"].values
longitudes = RAINFILE["longitude"].values
time_coords = RAINFILE["time"].values
nDays = int(np.size(RAINFILE.time.values) / tpd)  # Calculate number of days

# Convert watershed shapefile to raster mask and flip vertically to match NetCDF orientation
watershed_mask = np.flip(Shape2Array(WSShp, RAINFILE, True), axis=0)

# Find all valid data points (non-negative values) in the first time step
data = data_var.isel(time=0).values
rows, cols = np.where(data >= 0)
points = list(zip(rows, cols))            # List of (lat_idx, lon_idx) coordinates

# Get watershed mask dimensions
mask_shape = watershed_mask.shape
mask_rows, mask_cols = mask_shape

# Precompute indices of valid pixels within the watershed mask (value = 1)
valid_mask_indices = np.where(watershed_mask.flatten() == 1)[0]  # 1D indices for flattened mask

@njit(parallel=True)
def fast_masked_average(data_array, mask_indices, mask_shape, points):
    """
    JIT-compiled function for fast computation of watershed-averaged precipitation.
    Uses parallel processing to efficiently calculate masked averages for all points.
    
    Parameters:
    -----------
    data_array : numpy.ndarray
        3D array of precipitation data (time, lat, lon)
    mask_indices : numpy.ndarray
        1D indices of valid watershed pixels in flattened mask
    mask_shape : tuple
        Shape of the watershed mask (rows, cols)
    points : list
        List of (lat_idx, lon_idx) coordinates to process
        
    Returns:
    --------
    numpy.ndarray
        2D array of averaged values (points, time)
    """
    # Initialize results array with NaN values
    results = np.full((len(points), data_array.shape[0]), np.nan)
    
    mask_rows, mask_cols = mask_shape  # Extract mask dimensions

    # Process each point in parallel
    for i in prange(len(points)):
        lat_idx, lon_idx = points[i]
        
        # Define region bounds for placing the watershed mask
        lat_start, lat_end = lat_idx - mask_rows + 1, lat_idx + 1
        lon_start, lon_end = lon_idx, lon_idx + mask_cols 
        
        # Check if mask region is within data array bounds
        if lat_start < 0 or lat_end > data_array.shape[1] or lon_end > data_array.shape[2]:
            continue  # Skip points where mask would extend beyond data bounds

        # Calculate masked average for each time step
        for t in range(data_array.shape[0]):
            # Extract sub-region and flatten for indexing
            sub_data = data_array[t, lat_start:lat_end, lon_start:lon_end].flatten()
            masked_values = sub_data[mask_indices]  # Apply precomputed mask
            
            # Skip if any masked values are negative (invalid data)
            if np.min(masked_values) < 0:
                continue
                
            # Compute area-weighted average, ignoring NaN values
            results[i, t] = np.nanmean(masked_values)

    return results

# Convert xarray DataArray to NumPy array for efficient processing
data_np = data_var.values  # Shape: (time, lat, lon)

# Execute parallel watershed averaging computation
print("Computing watershed-averaged precipitation...")
masked_averages = fast_masked_average(data_np, valid_mask_indices, mask_shape, points)

# Initialize output array with same shape as input, filled with NaN
final_array = np.full_like(data_np, np.nan)

# Place computed watershed averages back into correct spatial locations
for i, (lat_idx, lon_idx) in enumerate(points):
    # Round to 1 decimal place and assign to all time steps for this location
    final_array[:, lat_idx, lon_idx] = np.round(masked_averages[i, :], decimals=1)

# Convert back to xarray DataArray with original coordinates and metadata
WAP = xr.DataArray(
    final_array, 
    coords={"time": time_coords, "latitude": latitudes, "longitude": longitudes}, 
    dims=["time", "latitude", "longitude"]
)

# Output daily WAP files
print("Writing daily WAP files...")
for bb in range(nDays):
    # Extract data for current day
    WAPDay = WAP.isel(time=slice(bb * tpd, (bb + 1) * tpd))
    
    # Format date string for filename (YYYYMMDD)
    date = WAPDay.time.values[0]
    formatted_date = date.astype('datetime64[D]').astype(str).replace("-", "")
    
    # Convert to dataset and save as compressed NetCDF4
    WAPDay = WAPDay.to_dataset(name="WAP")
    output_filename = f"WAP.{formatted_date}.{OutputFormat}.nc4"
    
    WAPDay.to_netcdf(
        output_filename,
        encoding={"WAP": {"zlib": True, "complevel": 9}},  # High compression
        format='NETCDF4_CLASSIC'
    )
    
print(f"Processing complete. Generated {nDays} daily WAP files.")

        
