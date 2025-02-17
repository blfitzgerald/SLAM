import xarray as xr
import sys

LMLoc=sys.argv[1]
LMNum=int(sys.argv[2])
dur=sys.argv[3]

LM=xr.open_dataset(LMLoc+"LM."+dur+"hr.0.nc")
for pp in range(1,LMNum):
    try:
        LMpp=xr.open_dataset(LMLoc+"LM."+dur+"hr."+str(pp)+".nc")

        LM=xr.merge([LM, LMpp])
    except:
        print('Missing '+str(pp))

LM.to_netcdf(LMLoc+"LMs."+dur+"hr.nc")