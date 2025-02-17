import xarray as xr
import sys

path=sys.argv[1]
dur=sys.argv[2]
precvar=sys.argv[3]
yearf=sys.argv[4]
yearl=sys.argv[5]

ds=xr.open_dataset(path+yearf+"."+str(dur)+"hrdur.AMWAP.nc4")
ds=ds.assign_coords({'year': yearf})

whenDS=xr.open_dataset(path+yearf+"."+str(dur)+"hrdur.AMWhen.nc4")
whenDS=whenDS.assign_coords({'year': yearf})

for aa in range(int(yearf)+1,int(yearl)+1):
	ds2=xr.open_dataset(path+str(aa)+"."+str(dur)+"hrdur.AMWAP.nc4")
	ds2=ds2.assign_coords({'year': aa})
	ds=xr.concat([ds,ds2], dim='year')

	whenDS2=xr.open_dataset(path+str(aa)+"."+str(dur)+"hrdur.AMWhen.nc4")
	whenDS2=whenDS2.assign_coords({'year': aa})
	whenDS=xr.concat([whenDS,whenDS2], dim='year')

whenDS= whenDS.rename({precvar: 'timeIndex'})
ds.to_netcdf(path+str(dur)+'hr.fullAM.nc4',encoding={precvar:{"zlib": True, "complevel": 9}},format='NETCDF4_CLASSIC')
whenDS.to_netcdf(path+str(dur)+'hr.AMtimeIndex.nc4',encoding={"timeIndex":{"zlib": True, "complevel": 9}},format='NETCDF4_CLASSIC')
