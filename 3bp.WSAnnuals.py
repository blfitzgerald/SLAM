def find_first_row_with_one(array):
    # Convert the array to a NumPy array
    array = np.array(array)

    # Find the indices of the first row with a 1
    row_indices = np.where(np.any(array == 1, axis=1))[0]

    if row_indices.size > 0:
        # Get the index of the first row with a 1
        first_row_index = row_indices[0]

        # Find the column index of the first 1 in the first row
        first_column_index = np.where(array[first_row_index] == 1)[0][0]

        # Return the full index of the first 1 in the first row
        return (first_row_index, first_column_index)
    else:
        # No row with a 1 found
        return None

import numpy as np
import xarray as xr
import fiona
from rasterio.io import MemoryFile
from rasterio.transform import from_origin
from rasterio.mask import mask
import sys
import math

dur=sys.argv[1]
precVar=sys.argv[2]
AMCLOC=sys.argv[3]
WSShp=sys.argv[4]
RAINLOC=sys.argv[5]
TIMESLOC=sys.argv[6]
precipType=sys.argv[7]

if precipType=='AORC':
    pp='AORC.'
    ps='.precip.nc'

if int(dur)<=25:
    nDays=2
else:
    nDays=math.ceil(int(dur)/24)+1

AMwhen=xr.open_dataset(AMCLOC+dur+"hr.AMtimeIndex.nc4")
AM=xr.open_dataset(AMCLOC+dur+"hr.fullAM.nc4")

latitude=np.array(AM.latitude)
longitude=np.array(AM.longitude)                                                            # Get all the latitudes and longitudes in the dataset

ydim=np.size(AM.latitude)
xdim=np.size(AM.longitude)

with fiona.open(WSShp, "r") as shapefile:
    shapes = [feature["geometry"] for feature in shapefile]

transform = from_origin(AM.longitude.isel(longitude=1),AM.latitude.isel(latitude=-1),np.abs(AM.longitude.isel(longitude=2)-AM.longitude.isel(longitude=1)) ,np.abs(AM.latitude.isel(latitude=2)-AM.latitude.isel(latitude=1)))

rastertemplate=np.ones((xdim,ydim),dtype='float32')

memfile = MemoryFile()
rastermask = memfile.open(driver='GTiff',
                          height = rastertemplate.shape[1], width = rastertemplate.shape[0],
                          count=1, dtype=str(rastertemplate.dtype),
                          crs='+proj=longlat +datum=NAD83 +no_defs',
                          transform=transform)

rastermask.write(rastertemplate,1)


simplemask, out_transform = mask(rastermask, shapes, crop=True,all_touched=True)
simplemask2, out_transform2 = mask(rastermask, shapes, crop=False,all_touched=True)

index1 = find_first_row_with_one(simplemask[0,:,:])

index2 = find_first_row_with_one(simplemask2[0,:,:])

siteLocIndex=np.array([index2[0]-index1[0],index2[1]-index1[1]])

WSLat=np.flip(AM.latitude.values, axis=0)[siteLocIndex[0]]
WSLon=AM.longitude.values[siteLocIndex[1]]

WSWhen=AMwhen.timeIndex.sel(latitude=WSLat, longitude=WSLon, method='nearest').values

WSAMs=AM[precVar].sel(latitude=WSLat, longitude=WSLon, method='nearest').values

WSArray=simplemask[0,:,:]
WSShape=np.shape(WSArray)

WSAnnual=np.zeros([WSShape[0],WSShape[1],43])

AMWS=AM[precVar].sel(latitude=WSLat, longitude=WSLon).values


for aa in range(np.size(WSWhen)):
    FileList=[]
    year=1979+aa
    times=np.load(TIMESLOC+str(year)+".times.npy")

    timeIndex=WSWhen[aa]
    
    hourLast=times[int(timeIndex)]
    hour=times[int(timeIndex)]
    hour_adj= hour - np.timedelta64(1, 'h')
    day=hour_adj.astype('datetime64[D]')
    daystr=str(day).replace('-','')
    FileList.append(RAINLOC+daystr[0:4]+"//"+pp+daystr+ps)

    for bb in range(1,nDays):
        hour=hour - np.timedelta64(24, 'h')
        hour_adj= hour - np.timedelta64(1, 'h')
        day=hour_adj.astype('datetime64[D]')
        daystr=str(day).replace('-','')

        
        FileList.append(RAINLOC+daystr[0:4]+"//"+pp+daystr+ps)
  

    yy1=np.where(latitude==WSLat)[0][0]
    xx1=np.where(longitude==WSLon)[0][0]
        

    lat2=latitude[yy1-WSShape[0]+1]
    lon2=longitude[xx1+WSShape[1]-1]

    yearRain=xr.open_mfdataset(FileList)

    AMFiles=yearRain.sel(time=slice(hour+np.timedelta64(1, 'h'),hourLast), latitude=slice(lat2,WSLat),longitude=slice(WSLon,lon2)).sum(dim='time').load()
    
    dayMasked=np.flip(WSArray,axis=0)*AMFiles

    WSAnnual[:,:,aa]=dayMasked[precVar].values

np.save(AMCLOC+"WS"+dur+"hrAMRecord.npy",WSAnnual)

        