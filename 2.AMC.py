## Last updated 11/11/2024
# Title: Annual Maxima Calculation
# Author: Benjamin FitzGerald
# Purpose: Calculates Annual Maxima for a year of data

# Inputs: 1) Duration of Precip, 2) The year testing on, 3) Name of Precip Variable,
# Inputs from code are compatible with execudable, AMC.sh, and submit file, AMC.sub though the sys.argv inputs could be replaced by the user. 
# Outputs: A netcdf files of the watershed average annual maxima and the timesteps they occur in

import xarray as xr 
import sys 
from netCDF4 import MFDataset

stormDuration=int(sys.argv[1])
year=sys.argv[2]
RAINVAR=sys.argv[3]

RAINFILE=xr.open_mfdataset(year+'*.WAP.nc4') 

# To calculate where AM is for a duration, use cumulative summation over the whole time, shift by the time, subtract off the shifted, this give the amount of precip that fell within the duration
# Then find when the max occurred. 

cumulative=RAINFILE.cumsum(dim='time') 
ccum=cumulative.copy(deep='True') 
SCC=ccum.shift(time=stormDuration) 
Diff=cumulative-SCC 
WSMax=Diff.max(dim='time') 
WIMax=Diff.argmax(dim='time',skipna=True)

WSMax=WSMax.round()

WSMax.to_netcdf(str(year)+"."+str(stormDuration)+"hrdur.AMWAP.nc4",encoding={RAINVAR:{"zlib": True, "complevel": 9}},format='NETCDF4') 
WIMax.to_netcdf(str(year)+"."+str(stormDuration)+"hrdur.AMWhen.nc4",encoding={RAINVAR:{"zlib": True, "complevel": 9}},format='NETCDF4')