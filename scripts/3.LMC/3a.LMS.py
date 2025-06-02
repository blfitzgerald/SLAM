## Last updated 3/11/2025
# Title: L-Moment Calculation Setup
# Author: Benjamin FitzGerald
# Purpose: The goal is to split our lmoment calculation into separate jobs to run in parallel based on their longitude value. So we split the points into groups, 
# then find the date with annual maxima precip for those points, then create input files that will submit the list of precip files and other necessary items to run the codes.. 

# Inputs: 1) Duration of Precip, 2) Where precipitation data is located, 3) Name of Precip Variable, 4) String for text before number in precip file, 5) string for text after the number in precip,
# 6) first year of precip data, 7) last year of precip data, 8) location of npy files that give what time a year's time index is, 9) Where the inputs for the LMC process will be (if not moved will be where outputs of this code go),
# 10) Where the LMC code is, 11) Where WSArray.npy is located, 12) Where the annual maxima files are located, 13/14) pyton environment for LMC, 15) what process this is from submit, 16) total number of jobs to break this into

# Inputs from code are compatible with execudable, LMS.sh, and submit file, LMS.sub though the sys.argv inputs could be replaced by the user. 
# Outputs: PointList$Process.npy, DateList$Process.npy, Input$Process.txt

import xarray as xr
import numpy as np
from numba import njit, prange
import math
import sys
import re

path=sys.argv[1]
AMKEY=sys.argv[2]
RAINKEY=sys.argv[3]

group_size=15

AM=xr.open_dataset(path+"AMs//"+AMKEY+"Maxima."+RAINKEY+".nc4")

latitudes=AM.latitude.values
longitudes=AM.longitude.values
yearlist=AM.year.values
yearf=int(yearlist[0])
nYears=np.size(AM.year.values)
yearl=int(yearlist[nYears-1])

nLats=np.size(latitudes)
nLons=np.size(longitudes)

amwap_array=AM.WAP.values
indices = np.where(amwap_array > 0)

years=indices[0]
lats=indices[1]
lons=indices[2]

lonlist=np.unique(np.sort(lons))
sorted_groups = [np.sort(lonlist[i:i+group_size]) for i in range(0, len(lonlist), group_size)]
num_groups=len(sorted_groups)

all_dates=AM.file_days.values

with open(path+"LMCInput."+RAINKEY+"."+AMKEY+'.txt', 'w') as infile: 

    for aa in range(num_groups):
        lon1=sorted_groups[aa][0]
        lon2=sorted_groups[aa][-1]
        lon_dates=all_dates[:,:,:,lon1:lon2+1]
        lon_dates=lon_dates.flatten()
        lon_dates=lon_dates[np.isnan(lon_dates)==False]
        uniqueDates=np.unique(lon_dates)
        dates=uniqueDates.astype(int)

        if np.size(dates)>0:
            if np.size(dates)>1000:
                ds=str(dates[0:1000]).replace('\n', '')
                counted=1000
                remaining=np.size(dates)-counted
                while remaining>0:
                    if remaining>=1000:
                        ds=ds+str(dates[counted:counted+1000]).replace('\n', '')
                        counted=counted+1000
                        remaining=np.size(dates)-counted

                    else: 
                        ds=ds+str(dates[counted:counted+remaining]).replace('\n', '')
                        counted=counted+remaining
                        remaining=np.size(dates)-counted
            else:
                ds=str(dates).replace('\n', '')
            ds = re.sub(r'\]\[', ' ', ds)
            ds=ds.replace('[','[$(PRECIPPATH)/YEAR/$(PRECIPPREFFIX)')       
            ds=ds.replace(' ','$(PRECIPSUFFIX), $(PRECIPPATH)/YEAR/$(PRECIPPREFFIX)')
            for yy in range(yearf,yearl+1):
                year=str(int(yy))
                ds=ds.replace('/YEAR/$(PRECIPPREFFIX)'+year,'/'+year+'/$(PRECIPPREFFIX)'+year)
            ds=ds.replace('[','')
            ds=ds.replace(']','$(PRECIPSUFFIX), $(CODELOC)/$(LMCCODE), $(DPLoc)/$(DPFile), $(WSLOC)/$(WSNAME).shp, $(WSLOC)/$(WSNAME).shx, $(AMLOC)/$(AMFILE), $(ENVLINK) \n')
            infile.write(ds)
      