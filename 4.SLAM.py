def remove_isolated_ones_from_index(arr, start_i, start_j):
    def is_valid(i, j):
        return 0 <= i < len(arr) and 0 <= j < len(arr[0]) and arr[i][j] == 1

    stack = [(start_i, start_j)]

    while stack:
        i, j = stack.pop()
        arr[i][j] = -1  # Mark visited

        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            ni, nj = i + dx, j + dy
            if is_valid(ni, nj):
                stack.append((ni, nj))

    # Remove isolated 1s
    for i in range(len(arr)):
        for j in range(len(arr[0])):
            if arr[i][j] == 1:
                arr[i][j] = 0
    return arr

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


import xarray as xr
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import lmoments3 as lm3
import statsmodels.stats.multitest
from scipy.stats import percentileofscore
from scipy.stats import moment, skew, kurtosis
import sys
import fiona
from rasterio.io import MemoryFile
from rasterio.transform import from_origin
from rasterio.mask import mask
from matplotlib import pyplot as plt
import geopandas as gpd
from shapely.geometry import Polygon, MultiPolygon

dur=sys.argv[1]
gp=float(sys.argv[2])
WSShp=sys.argv[3]+'.shp'
rainVar=sys.argv[4]

# Open the L-Moment grid values and the annual maxima record for the watershed
LM=xr.open_dataset("LMs."+dur+"hr.nc")
WSAnnuals=np.load("WS"+dur+"hrAMRecord.npy")
ntrials=10000

WSShape=np.shape(WSAnnuals)

YearList=np.random.randint(0,WSShape[2]-1, size=[WSShape[2],ntrials])
lmoms=np.zeros([5,ntrials])

# Run the bootstrap method- randomly sample years from watershed record, take mean across time, then calculate L-moments
for tt in range(ntrials):
    yearsIndices=YearList[:,tt]
    runSum=np.zeros([WSShape[0],WSShape[1]])
    
    for yy in range(np.size(yearsIndices)):
        rain=WSAnnuals[:,:,yearsIndices[yy]]
        
        runSum=runSum+rain
            
    flattened_list = runSum.flatten()
    filtered_list = flattened_list[~np.isnan(flattened_list)].tolist()
    filtered_list = [element for element in filtered_list if element != 0]
    filtered_list = np.array(filtered_list) / np.size(yearsIndices)
    lmoms[0:4,tt]=lm3.lmom_ratios(filtered_list,4)
    lmoms[4,tt]=lmoms[1,tt]/lmoms[0,tt]

L1Test=lmoms[0,:]
LSkewTest=lmoms[2,:]
LKurtTest=lmoms[3,:]
LCVTest=lmoms[4,:]


np.save('L1Test.'+str(dur)+'hr.npy', L1Test)
np.save('LSkewTest.'+str(dur)+'hr.npy', LSkewTest)
np.save('LKurtTest.'+str(dur)+'hr.npy', LKurtTest)
np.save('LCVTest.'+str(dur)+'hr.npy', LCVTest)


# For analysis, transform the Lmoment arrays into flattened arrays for each with corresponding lat and lon stored. 

latitude=np.array(LM.latitude)
longitude=np.array(LM.longitude)  

TDLatArray=np.zeros([np.size(latitude),np.size(longitude)])
TDLonArray=np.zeros([np.size(latitude),np.size(longitude)])

for aa in range(np.size(latitude)):
     for bb in range(np.size(longitude)):
         TDLatArray[aa,bb]=latitude[aa]
         TDLonArray[aa,bb]=longitude[bb]

L1Flat=LM.l1.values.reshape([np.size(longitude)*np.size(latitude)])
LCVFlat=LM.lcv.values.reshape([np.size(longitude)*np.size(latitude)])
LSkewFlat=LM.l3.values.reshape([np.size(longitude)*np.size(latitude)])
LKurtFlat=LM.l4.values.reshape([np.size(longitude)*np.size(latitude)])

TDLat=TDLatArray.reshape(np.size(longitude)*np.size(latitude))
TDLon=TDLonArray.reshape(np.size(longitude)*np.size(latitude))

non_nan_columns = np.isnan(LCVFlat)

TDLat= TDLat[np.where(non_nan_columns==False)]
TDLon= TDLon[np.where(non_nan_columns==False)]

TDL1Flat= L1Flat[np.where(non_nan_columns==False)]
TDLCVFlat= LCVFlat[np.where(non_nan_columns==False)]
TDLSkewFlat= LSkewFlat[np.where(non_nan_columns==False)]
TDLKurtFlat= LKurtFlat[np.where(non_nan_columns==False)]

# Once flattened, calculate the p-value of each location's L-moments against the bootstrap test values
testSamples=np.size(TDLon)
L1p=np.zeros(testSamples)
L3p=np.zeros(testSamples)
L4p=np.zeros(testSamples)
LCVp=np.zeros(testSamples)

for pp in range(testSamples):
    L1=TDL1Flat[pp]
    L3=TDLSkewFlat[pp]
    L4=TDLKurtFlat[pp]
    LCV=TDLCVFlat[pp]
    
    L1p[pp]=0.5-np.abs((percentileofscore(L1Test, L1)-50)/100)
    L3p[pp]=0.5-np.abs((percentileofscore(LSkewTest, L3)-50)/100)
    L4p[pp]=0.5-np.abs((percentileofscore(LKurtTest, L4)-50)/100)
    LCVp[pp]=0.5-np.abs((percentileofscore(LCVTest, LCV)-50)/100)

# Run false discovery rate loops, which will remove worst points until all points pass. One loop for each l-moment
maxed=0
L1p1=np.copy(L1p)
L1Lon=np.copy(TDLon)
L1Lat=np.copy(TDLat)

while maxed==0:
    domainSize=np.size(L1p1)
    L1Reject,L1Pval=statsmodels.stats.multitest.fdrcorrection(L1p1, alpha=gp, method='poscorr', is_sorted=False)
    sigSize=domainSize-np.sum(L1Reject)

    if domainSize==sigSize:
        maxed=1
    else:
        minVal=np.argmin(L1p1)
        L1p1=np.delete(L1p1, minVal)
        L1Lon=np.delete(L1Lon, minVal)
        L1Lat=np.delete(L1Lat, minVal)

print('Done with L1')   

maxed=0
L3p3=np.copy(L3p)
L3Lon=np.copy(TDLon)
L3Lat=np.copy(TDLat)

while maxed==0:
    domainSize=np.size(L3p3)
    L3Reject,L3Pval=statsmodels.stats.multitest.fdrcorrection(L3p3, alpha=gp, method='poscorr', is_sorted=False)
    sigSize=domainSize-np.sum(L3Reject)

    if domainSize==sigSize:
        maxed=1
    else:
        minVal=np.argmin(L3p3)
        L3p3=np.delete(L3p3, minVal)
        L3Lon=np.delete(L3Lon, minVal)
        L3Lat=np.delete(L3Lat, minVal)

print('Done with L3')   

maxed=0
L4Lon=np.copy(TDLon)
L4Lat=np.copy(TDLat)
L4p4=np.copy(L4p)

while maxed==0:
    domainSize=np.size(L4p4)
    L4Reject,L4Pval=statsmodels.stats.multitest.fdrcorrection(L4p4, alpha=gp, method='poscorr', is_sorted=False)
    sigSize=domainSize-np.sum(L4Reject)

    if domainSize==sigSize:
        maxed=1
    else:
        minVal=np.argmin(L4p4)
        
        L4p4=np.delete(L4p4, minVal)
                
        L4Lon=np.delete(L4Lon, minVal)
        L4Lat=np.delete(L4Lat, minVal)

print('Done with L4')   

maxed=0
LCVLon=np.copy(TDLon)
LCVLat=np.copy(TDLat)
LCVpCV=np.copy(LCVp)

while maxed==0:
    domainSize=np.size(LCVpCV)
    LCVReject,LCVPval=statsmodels.stats.multitest.fdrcorrection(LCVpCV, alpha=gp, method='poscorr', is_sorted=False)
    sigSize=domainSize-np.sum(LCVReject)

    if domainSize==sigSize:
        maxed=1
    else:
        minVal=np.argmin(LCVpCV)
        
        LCVpCV=np.delete(LCVpCV, minVal)
        
        
        LCVLon=np.delete(LCVLon, minVal)
        LCVLat=np.delete(LCVLat, minVal)

print('Done with LCV')   

PassFlags=np.zeros([np.size(latitude),np.size(longitude)])
L1Result=np.zeros([np.size(latitude),np.size(longitude)])
LCVResult=np.zeros([np.size(latitude),np.size(longitude)])
L3Result=np.zeros([np.size(latitude),np.size(longitude)])
L4Result=np.zeros([np.size(latitude),np.size(longitude)])

# Go through lists and find where each of the footprints passed for each l-moment
for aa in range(np.size(L1Lon)):
      xx=np.where(longitude==L1Lon[aa])
      yy=np.where(latitude==L1Lat[aa])
    
      PassFlags[yy,xx]=PassFlags[yy,xx]+1
      L1Result[yy,xx]=1

for aa in range(np.size(LCVLon)):
      xx=np.where(longitude==LCVLon[aa])
      yy=np.where(latitude==LCVLat[aa])
    
      PassFlags[yy,xx]=PassFlags[yy,xx]+1
      LCVResult[yy,xx]=1

for aa in range(np.size(L3Lon)):
      xx=np.where(longitude==L3Lon[aa])
      yy=np.where(latitude==L3Lat[aa])
    
      PassFlags[yy,xx]=PassFlags[yy,xx]+1
      L3Result[yy,xx]=1

for aa in range(np.size(L4Lon)):
      xx=np.where(longitude==L4Lon[aa])
      yy=np.where(latitude==L4Lat[aa])
    
      PassFlags[yy,xx]=PassFlags[yy,xx]+1
      L4Result[yy,xx]=1

dims = ("latitude", "longitude")
coords = {
    "latitude": latitude,
    "longitude": longitude
}

PassingPts = xr.Dataset(data_vars=dict(PassFlag=(['latitude','longitude'],PassFlags), L1=(['latitude','longitude'],L1Result), LCV=(['latitude','longitude'], LCVResult), L3=(['latitude','longitude'],L3Result), L4=(['latitude','longitude'],L4Result)),
    coords=coords
)
    
PassingPts.to_netcdf(str(dur)+'Hr.'+str(gp)+"PassingScores.nc")

Results=PassingPts
output=dur+'hr.'+str(gp)+"Domain.shp"

Results=Results.where(Results.PassFlag==4)
Results['PassFlag']=Results.PassFlag/Results.PassFlag

# Extract some useful info from our data:
latitude=np.array(Results.latitude)
longitude=np.array(Results.longitude) 
ydim=np.size(latitude)
xdim=np.size(longitude)

WSArray=Shape2Array(WSShp, Results, True)
WSShape=np.shape(WSArray)

WSFull=Shape2Array(WSShp, Results, False)

regionLat=Results.latitude
regionLon=Results.longitude
regionLat=np.flip(regionLat)

dx=regionLon[1]-regionLon[0]
dy=regionLat[1]-regionLat[0]

regionX=np.size(regionLon)
regionY=np.size(regionLat)

passArray=np.flip(Results.PassFlag.values,axis=0)
TestDomain=Results.PassFlag.values
passArray2=remove_isolated_ones_from_index(passArray,np.min(np.where(WSFull==1)[0]),np.min(np.where(WSFull==1)[1]))
testCount=np.zeros([np.size(regionLat)+WSShape[0],np.size(regionLon)])

passCount=np.zeros([np.size(regionLat),np.size(regionLon)])

for aa in range(regionY):
    for bb in range(regionX):
        if TestDomain[aa,bb]==1:
            testCount[aa:aa+WSShape[0],bb:bb+WSShape[1]]=testCount[aa:aa+WSShape[0],bb:bb+WSShape[1]]
        if passArray2[aa,bb]==-1:
            passCount[aa:aa+WSShape[0],bb:bb+WSShape[1]]=passCount[aa:aa+WSShape[0],bb:bb+WSShape[1]]+WSArray

newX=np.shape(passCount)[1]
newY=np.shape(passCount)[0]

newLat=np.zeros(newY)
newLon=np.zeros(newX)

for aa in range(newY):
    newLat[aa]=regionLat[-1]-aa*dy
    
#newLat=np.flip(newLat)
for bb in range(newX):
    newLon[bb]=regionLon[0]+bb*dx
    
dims = ("latitude", "longitude")
coords = {
    "latitude": np.flip(regionLat,axis=0),
    "longitude": regionLon
}

ds = xr.Dataset(data_vars=dict(passed=(['latitude','longitude'],np.flip(passCount, axis=0))),
    coords=coords
)

# Specify the variable you want to pad
values_var = ds['passed']  # Replace with your variable name

# Get the original latitude and longitude coordinates
original_latitude = values_var.latitude
original_longitude = values_var.longitude

# Define the padding width (1 row of zeros)
padding_width = 1

# Create a new xarray dataset with padded values
padded_values = np.pad(values_var, ((padding_width, padding_width), (padding_width, padding_width)), mode='constant')

# Adjust the latitude and longitude coordinates
padded_latitude = np.linspace(original_latitude[0] - padding_width*dy, original_latitude[-1] + padding_width*dy, len(original_latitude) + 2 * padding_width)
padded_longitude = np.linspace(original_longitude[0] - padding_width*dx, original_longitude[-1] + padding_width*dx, len(original_longitude) + 2 * padding_width)

# Create a new xarray dataset with the padded and adjusted values
padded_ds = xr.Dataset({'passed': (['latitude', 'longitude'], padded_values)},
                       coords={'latitude': padded_latitude, 'longitude': padded_longitude})

# Save the padded dataset to a new NetCDF

# Specify the variable with the values
values_var = padded_ds['passed']  # Replace with your variable name

# Define the contour level above zero
contour_level = 0.0  # Adjust as needed

# Create a plot with contour lines for values above zero
plt.figure(figsize=(8, 8))
contours = plt.contour(padded_ds.longitude, padded_ds.latitude, values_var, levels=[contour_level])

paths=contours.collections[0].get_paths()

# Create a list to hold polygons
polygons = []

# Iterate over paths to construct Polygons with holes
for path in paths:
    # Get vertices of the current path
    vertices = path.vertices

    # Check if there are subpaths (holes)
    if len(path.to_polygons()) > 1:
        # Separate the outer boundary (shell) and inner boundaries (holes)
        shell = path.to_polygons()[0]  # Outer boundary
        holes = path.to_polygons()[1:]  # Inner boundaries
        polygons.append(Polygon(shell=shell, holes=holes))
    else:
        # Only an outer boundary exists, no holes
        polygons.append(Polygon(shell=vertices))

# Create a GeoDataFrame with the Polygons
gdf = gpd.GeoDataFrame(geometry=polygons)

# Set the CRS to WGS84 (EPSG:4326)
gdf.crs = "EPSG:4326"

gdf.to_file(output)







