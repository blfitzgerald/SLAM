## Last updated 1/29/2025
# Title: Point Precipitation to Watershed Precipitation
# Author: Benjamin FitzGerald
# Purpose: Checks that the test domain area has the correct number of points with precipitation data and the calculates watershed totals
#
# Inputs: 1) Name of watershed shapefile to open, 2) Name of test domain shapefile to open, 3) Name of Precip Variable,
# 4) Name of Longitude Dimension, 5) Name of Latitude Dimension, 6) Timesteps per daily file
# Inputs from code are compatible with execudable, PP2WAP.sh, and submit file, PP2WAP.sub though the sys.argv inputs could be replaced by the user. 
# Outputs: A netcdf file that indicates the watershed average precipitation for each transposed shape, a numpy array called WSArray.npy that is an array of the watershed for the precipitation data grid

from rasterio.io import MemoryFile
from rasterio.transform import from_origin
from rasterio.mask import mask
import fiona
import numpy as np
import xarray as xr 
from netCDF4 import Dataset,date2num
import sys

def Shape2Array(shp, rast, crop):
    
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
TDShp= sys.argv[2]+".shp"
precvar=sys.argv[3]
lonName=sys.argv[4]									# Longitude dimension name
latName=sys.argv[5]									# Latitude dimension name
tpd=int(sys.argv[6])

# First we open all the precipitation files that we've transferred to the file
RAINFILE = xr.open_mfdataset('*.nc').load()

# Rename dimensions if not properly named. 
if 'longitude' not in RAINFILE.dims:
    RAINFILE=RAINFILE.rename({lonName:'longitude'})
if 'latitude' not in RAINFILE.dims:
    RAINFILE=RAINFILE.rename({latName:'latitude'})

nDays=int(np.size(RAINFILE.time.values)/tpd)

PRCPMask=RAINFILE.isel(time=0).where(RAINFILE[precvar].isel(time=0)>=0)

PRCPMask[precvar]=(PRCPMask[precvar]+1)/(PRCPMask[precvar]+1)
TestDomainMask=PRCPMask.copy(deep=True)
TestDomain=PRCPMask.copy(deep=True)
TestDomain=TestDomain.where(TestDomain[precvar]>1)


WAP=RAINFILE.copy(deep=True)
WAP=WAP*0

latitude=RAINFILE.latitude.values
longitude=RAINFILE.longitude.values
ydim=np.size(latitude)
xdim=np.size(longitude)

shape=np.flip(Shape2Array(WSShp, RAINFILE, False),axis=0)
shape[shape==0]=np.nan
wsAM=RAINFILE.isel(time=0)*shape

wsAMArray=wsAM.precrate.values

arr=wsAMArray

WSmask = ~np.isnan(arr) 

# Step 2: Find the bounding box indices
rows = np.any(WSmask, axis=1)  # Rows with at least one valid value
cols = np.any(WSmask, axis=0)  # Columns with at least one valid value

# Get the min/max row and column indices
rmin, rmax = np.where(rows)[0][[0, -1]]  # First and last row with valid values
cmin, cmax = np.where(cols)[0][[0, -1]]  # First and last column with valid values

# Step 3: Slice the array using the bounding box indices
WSArray= wsAMArray[rmin:rmax+1, cmin:cmax+1]

WSArray[WSArray==0]=1
WSArray[WSArray!=1]=0

WSArray=np.flip(WSArray, axis=0)
WSShape=np.shape(WSArray)
np.save("WSArray.npy", WSArray)

TDArray=Shape2Array(TDShp, RAINFILE, False)
TDShape=np.shape(TDArray)

TestDomainMask=TestDomainMask*np.flip(TDArray,axis=0)
TestDomainMask=TestDomainMask.where(TestDomainMask[precvar]>0)

data=TestDomainMask[precvar].values
rows, cols = np.where(data == 1)
min_row, max_row = rows.min(), rows.max()
min_col, max_col = cols.min(), cols.max()

startx=min_col-WSShape[1]
starty=min_row
endx=max_col
endy=max_row+WSShape[0]

if startx<0:
    startx=0

if starty<WSShape[0]:
    starty=WSShape[0]

if endx+WSShape[1]>xdim:
    endx=xdim-WSShape[1]

if endy>ydim:
    endy=ydim

print(xdim)
print(startx)
print(endx)
print(ydim)
print(starty)
print(endy)
for xx in range(startx, endx):
    for yy in range(starty, endy):
    
        lat=float(PRCPMask.latitude.isel(latitude=yy).values)
        lon=float(PRCPMask.longitude.isel(longitude=xx).values)
        
        movingmask=np.zeros([ydim,xdim])
        
        print(xx)
        print(yy)
        print(WSShape)
        print(yy-WSShape[0])
        print(xx+WSShape[1])
        movingmask[yy-WSShape[0]:yy,xx:xx+WSShape[1]]=np.flip(WSArray,axis=0)
        
        dayMasked=movingmask*RAINFILE
        WSSum=dayMasked.sum(dim=['latitude', 'longitude'])
        
        TDFP=movingmask*TestDomainMask
        TDSum=TDFP[precvar].sum().values
        
        
        if TDSum>0 and WSSum[precvar].min().values>=0:                                #Lei: why do we check the WSSum and also TDSum? is there case WSSum >0 but TDSum<=0?
            WAP.loc[dict(latitude=lat, longitude=lon)] = WSSum/np.sum(WSArray)

for bb in range(nDays):
    WAPDay=WAP.isel(time=slice(bb*tpd,(bb+1)*tpd))
    date=WAPDay.time.values[0]
    formatted_date = date.astype('datetime64[D]').astype(str).replace("-", "")
    
    WAPDay.to_netcdf(formatted_date+'.WAP.nc4',encoding={precvar:{"zlib": True, "complevel": 9}},format='NETCDF4_CLASSIC')

        
