import xarray as xr
import sys

LMLoc=sys.argv[1]
AMKEY=sys.argv[2]
FILEKEY1=sys.argv[3]
FILEKEY2=sys.argv[4]

AM=xr.open_dataset(LMLoc+"/AMs//"+AMKEY+"Maxima."+FILEKEY1+".nc4")

LM=xr.open_mfdataset(LMLoc+"/LMs/LMCol.*."+FILEKEY1+'.'+AMKEY+'.'+FILEKEY2+'.nc')

LM=LM.reindex(longitude=AM.longitude)
    
LM.to_netcdf(LMLoc+"/LMs/LMs."+FILEKEY1+'.'+AMKEY+'.'+FILEKEY2+".nc")