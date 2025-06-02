import xarray as xr
import sys

AMCKEY=sys.argv[1]
FILEKEY=sys.argv[2]

ds=xr.open_mfdataset(AMCKEY+"Maximum.*."+FILEKEY+".nc4")

ds.to_netcdf(AMCKEY+"Maxima."+FILEKEY+".nc4",encoding={"WAP":{"zlib": True, "complevel": 9}, 'file_days':{"zlib": True, "complevel": 9}, 'WAM_start':{"zlib": True, "complevel": 9}, 'WAM_end':{"zlib": True, "complevel": 9}},format='NETCDF4')

