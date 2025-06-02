import xarray as xr
import numpy as np
from numba import njit, prange
import fiona
from rasterio.io import MemoryFile
from rasterio.transform import from_origin
from rasterio.mask import mask
import sys

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

WSShp= sys.argv[1]+".shp"								# Watershed Shapefile 
precvar=sys.argv[2]
lonName=sys.argv[3]									# Longitude dimension name
latName=sys.argv[4]									# Latitude dimension name
tpd=int(sys.argv[5])
OutputFormat=sys.argv[6]

# First we open all the precipitation files that we've transferred to the file
RAINFILE = xr.open_mfdataset('*.nc').load()
if 'longitude' not in RAINFILE.dims:
    RAINFILE=RAINFILE.rename({lonName:'longitude'})
if 'latitude' not in RAINFILE.dims:
    RAINFILE=RAINFILE.rename({latName:'latitude'})

data_var = RAINFILE[precvar]  # Select the variable of interest
latitudes = RAINFILE["latitude"].values
longitudes = RAINFILE["longitude"].values
time_coords = RAINFILE["time"].values
nDays=int(np.size(RAINFILE.time.values)/tpd)

# Load the watershed mask (assume it's a NumPy array with shape (M, N))
watershed_mask = np.flip(Shape2Array(WSShp, RAINFILE, True),axis=0)

# Define list of (lat_idx, lon_idx) where you want to place the mask
data=data_var.isel(time=0).values
rows, cols = np.where(data>= 0)

points = list(zip(rows, cols))

# Get the mask size
mask_shape = watershed_mask.shape
mask_rows, mask_cols = mask_shape

# Precompute valid indices inside the mask
valid_mask_indices = np.where(watershed_mask.flatten() == 1)[0]  # 1D indices

# JIT-compiled function for fast averaging with parallelization
@njit(parallel=True)
def fast_masked_average(data_array, mask_indices, mask_shape, points):
    results = np.full((len(points), data_array.shape[0]), np.nan)  # Store results (points, time)
    
    mask_rows, mask_cols = mask_shape  # Extract mask dimensions

    for i in prange(len(points)):  # Parallelize the loop over points
        lat_idx, lon_idx = points[i]
        
        # Extract region around point
        lat_start, lat_end = lat_idx-mask_rows+1, lat_idx+1
        lon_start, lon_end = lon_idx, lon_idx+mask_cols 
        
        # Check bounds
        if lat_start<0 or lat_end > data_array.shape[1] or lon_end > data_array.shape[2]:
            continue  # Skip points where mask would be out of bounds

        # Apply mask and compute mean for each timestep
        for t in range(data_array.shape[0]):
            sub_data = data_array[t, lat_start:lat_end, lon_start:lon_end].flatten()
            masked_values = sub_data[mask_indices]  # Use precomputed mask indices
            if np.min(masked_values)<0:
                continue
            results[i, t] = np.nanmean(masked_values)  # Compute mean, ignoring NaNs

    return results

# Convert xarray data variable to NumPy array for fast processing
data_np = data_var.values  # Shape (time, lat, lon)

# Run the optimized function with parallelization
masked_averages = fast_masked_average(data_np, valid_mask_indices, mask_shape, points)  # Shape (points, time)

# Create an empty array matching the original shape (time, lat, lon)
final_array = np.full_like(data_np, np.nan)  # Same shape, filled with NaNs

# Place the computed values back into the correct locations
for i, (lat_idx, lon_idx) in enumerate(points):
    final_array[:, lat_idx, lon_idx] = np.round(masked_averages[i, :], decimals=1)  # Assign values for each time step

# Convert back to an xarray DataArray with original coordinates
WAP = xr.DataArray(final_array, coords={"time": time_coords, "latitude": latitudes, "longitude": longitudes}, 
                         dims=["time", "latitude", "longitude"])

for bb in range(nDays):
    WAPDay=WAP.isel(time=slice(bb*tpd,(bb+1)*tpd))
    date=WAPDay.time.values[0]
    formatted_date = date.astype('datetime64[D]').astype(str).replace("-", "")
    WAPDay=WAPDay.to_dataset(name="WAP")
    WAPDay.to_netcdf("WAP."+formatted_date+'.'+OutputFormat+'.nc4',encoding={"WAP":{"zlib": True, "complevel": 9}},format='NETCDF4_CLASSIC')

        
