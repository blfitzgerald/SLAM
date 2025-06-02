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
import numpy as np
import math
import pandas as pd
from datetime import datetime, timedelta

def get_file_date_str(timestamp: np.datetime64) -> str:
    """
    Given a datetime64[ns] timestamp, return the date (YYYYMMDD string)
    of the file the hour belongs to. Each file includes 01:00 to 00:00.
    """
    ts = pd.to_datetime(timestamp)
    if ts.hour == 0:
        file_date = ts.normalize() - pd.Timedelta(days=1)
    else:
        file_date = ts.normalize()
    return file_date.strftime("%Y%m%d")

def date_range_strings(start_date_str: str, end_date_str: str) -> list:
    """
    Given two dates in 'YYYYMMDD' format, return a list of all dates
    between them (inclusive) as 'YYYYMMDD' strings.
    """
    start_date = datetime.strptime(start_date_str, "%Y%m%d")
    end_date = datetime.strptime(end_date_str, "%Y%m%d")

    if end_date < start_date:
        raise ValueError("End date must be on or after start date.")

    date_list = []
    current_date = start_date
    while current_date <= end_date:
        date_list.append(current_date.strftime("%Y%m%d"))
        current_date += timedelta(days=1)

    return date_list

stormDuration=int(sys.argv[1])
year=sys.argv[2]
RAINVAR='WAP'
start_date = year+sys.argv[3] 
end_date = year+sys.argv[4]     

amkey=sys.argv[5]
outkey=sys.argv[6]

RAINFILE = xr.open_mfdataset('*.nc4')


# --- NEW: Apply time range filtering if seasonal dates are provided ---
start_dt = pd.to_datetime(start_date)
end_dt = pd.to_datetime(end_date) + pd.Timedelta(hours=23)  # inclusive of the end date
RAINFILE = RAINFILE.sel(time=slice(start_dt, end_dt))

if int(stormDuration)<=25:
    nDays=2
else:
    nDays=math.ceil(int(stormDuration)/24)+1

# To calculate where AM is for a duration, use cumulative summation over the whole time, shift by the time, subtract off the shifted, this give the amount of precip that fell within the duration
# Then find when the max occurred. 

cumulative=RAINFILE.cumsum(dim='time', skipna=False) 
ccum=cumulative.copy(deep='True') 
SCC=ccum.shift(time=stormDuration) 
Diff=cumulative-SCC 
WSMax=Diff.max(dim='time').compute()
#WSMax=WSMax.round()

WIMax=Diff.isel(time=slice(stormDuration,np.size(Diff.time.values))).argmax(dim='time', skipna=False).compute()
WIMax=WIMax.where(WSMax.WAP.notnull())

WIMax=WIMax.rename({"WAP":"timeIndex"})


TList=RAINFILE["time"].values

TIs=WIMax.timeIndex.values
indices = np.where(np.isnan(TIs[:,:])==False)
dates_array = np.full((nDays,TIs.shape[0],TIs.shape[1]), np.nan)
am_start_array=np.full((TIs.shape[0],TIs.shape[1]), np.nan)
am_end_array=np.full((TIs.shape[0],TIs.shape[1]), np.nan)

for aa in range(np.shape(indices)[1]):
     
    TI= TIs[indices[0][aa], indices[1][aa]]+1
    date=TList[int(TI)-int(stormDuration)]
    
    am_start_array[indices[0][aa], indices[1][aa]]=TList[int(TI)]
    am_end_array[indices[0][aa], indices[1][aa]]=TList[int(TI)+stormDuration-1]
    
    start=get_file_date_str(TList[int(TI)-1])
    end=get_file_date_str(TList[int(TI)-1+stormDuration])

    dates = date_range_strings(start, end)
    if np.size(dates)==nDays:    
        dates_array[:,indices[0][aa], indices[1][aa]]=dates
    else:
        dates_array[0:np.size(dates),indices[0][aa], indices[1][aa]]=dates

AM=WSMax        
AM['file_days']=(['nDays','latitude','longitude'],dates_array)
AM['WAM_start']=(['latitude','longitude'], am_start_array)
AM['WAM_end']=(['latitude','longitude'], am_end_array)

AM=AM.expand_dims(dim={"year":[int(year)]})
AM.to_netcdf(amkey+"Maximum."+str(year)+"."+outkey+".nc4",encoding={RAINVAR:{"zlib": True, "complevel": 9}, 'file_days':{"zlib": True, "complevel": 9}, 'WAM_start':{"zlib": True, "complevel": 9}, 'WAM_end':{"zlib": True, "complevel": 9}},format='NETCDF4')