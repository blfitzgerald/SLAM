## Last updated 11/11/2024
# Title: L-Moment Calculation Setup
# Author: Benjamin FitzGerald
# Purpose: The goal is to split our lmoment calculation into ~2000 separate jobs to run in parallel. So we split the points into groups, 
# then find the date with annual maxima precip for those points, then create input files that will submit the list of points, dates, and other necessary items. 

# Inputs: 1) Duration of Precip, 2) Where precipitation data is located, 3) Name of Precip Variable, 4) String for text before number in precip file, 5) string for text after the number in precip,
# 6) first year of precip data, 7) last year of precip data, 8) location of npy files that give what time a year's time index is, 9) Where the inputs for the LMC process will be (if not moved will be where outputs of this code go),
# 10) Where the LMC code is, 11) Where WSArray.npy is located, 12) Where the annual maxima files are located, 13/14) pyton environment for LMC, 15) what process this is from submit, 16) total number of jobs to break this into

# Inputs from code are compatible with execudable, LMS.sh, and submit file, LMS.sub though the sys.argv inputs could be replaced by the user. 
# Outputs: PointList$Process.npy, DateList$Process.npy, Input$Process.txt

import xarray as xr
import numpy as np
import re
import math
import sys
import warnings

warnings.filterwarnings("ignore")

dur=sys.argv[1]
RAINLOC=sys.argv[2]
rainVar=sys.argv[3]
RAINKEY=sys.argv[4]
RAINPOST=sys.argv[5]
yearf=int(sys.argv[6])
yearl=int(sys.argv[7])
TIMESLOC=sys.argv[8]
LMINPUT=sys.argv[9]
LMCCODE=sys.argv[10]
WSARRAYLOC=sys.argv[11]
AMLOC=sys.argv[12]
ENVLINK=sys.argv[13]
envname=sys.argv[14]
process=int(sys.argv[15])
total=int(sys.argv[16])

if int(dur)<=25:
    nDays=2
else:
    nDays=math.ceil(int(dur)/24)+1

AMwhenMasked=xr.open_dataset(dur+"hr.AMtimeIndex.nc4")
AMMasked=xr.open_dataset(dur+"hr.fullAM.nc4")

AMwhenMasked['year'] = AMwhenMasked['year'].astype(int)
AMMasked['year'] = AMMasked['year'].astype(int)

nYears=np.size(AMwhenMasked.year.values)

indices = np.argwhere(AMMasked[rainVar].sum(dim='year').values>0)
nPoints=np.shape(indices)[0]

runPoints=math.ceil(nPoints/total)

print(nPoints)
latitudeMask=AMMasked.latitude.values
longitudeMask=AMMasked.longitude.values

maxrun=(process*runPoints)+runPoints

FileList=np.zeros([runPoints,nDays,nYears])
PointList=np.zeros([runPoints,2])

if maxrun>=nPoints:
    Points2Run=runPoints-(maxrun-nPoints)
    
    if Points2Run<=0:
        sys.exit()
    maxrun=nPoints
    FileList=np.zeros([Points2Run,nDays,nYears])
count=0

for pp in range(process*runPoints, maxrun): 
    print(pp)
    xx=indices[pp,1]
    yy=indices[pp,0]
    
    lat=float(latitudeMask[yy])
    lon=float(longitudeMask[xx])
    
    PointList[count,0]=lat
    PointList[count,1]=lon
    
    for aa in range(yearl-yearf+1):
        year=aa+yearf
        times=np.load(str(year)+".times.npy")
        timeIndices=AMwhenMasked.timeIndex.sel(year=year, longitude=lon, latitude=lat, method='nearest').values
        uniqueTimes=np.unique(timeIndices)
        
        hour=times[int(uniqueTimes)]
        hour_adj= hour - np.timedelta64(1, 'h')
        day=hour_adj.astype('datetime64[D]')
        FileList[count,0,aa]=int(str(day).replace('-',''))

        for bb in range(1,nDays):
            hour=hour - np.timedelta64(24, 'h')
            hour_adj= hour - np.timedelta64(1, 'h')
            day=hour_adj.astype('datetime64[D]')
        
        
            FileList[count,bb,aa]=int(str(day).replace('-',''))
    count=count+1
    
np.save('PointList.'+str(dur)+'hr.'+str(process)+'.npy', PointList)

flatDateList=FileList.reshape([np.shape(FileList)[0]*np.shape(FileList)[1]*np.shape(FileList)[2]])
uniqueDates = np.unique(flatDateList)

with open("Input."+str(dur)+'hr.'+str(process)+".txt", 'w') as infile:
        dates=uniqueDates.astype(int)
        ds=str(dates).replace('\n', '')
        ds=ds.replace('[','[$(PRECIPPATH)/YEAR/$(PRECIPPREFFIX)')       
        ds=ds.replace(' ','$(PRECIPSUFFIX), $(PRECIPPATH)/YEAR/$(PRECIPPREFFIX)')
        for yy in range(yearf,yearl+1):
            year=str(int(yy))
            ds=ds.replace('/YEAR/$(PRECIPPREFFIX)'+year,'/'+year+'/$(PRECIPPREFFIX)'+year)
        ds=ds.replace('[','$(LMINPUT)DateList.'+str(dur)+'hr.'+str(process)+'.npy, $(LMINPUT)PointList.'+str(dur)+'hr.'+str(process)+'.npy, ')
        ds=ds.replace(']','$(PRECIPSUFFIX), $(LMCCODE), $(WSARRAYLOC)/WSArray.npy, $(AMLOC)'+str(dur)+'hr.AMtimeIndex.nc4, $(AMLOC)'+str(dur)+'hr.fullAM.nc4, $(TIMESLOC), $(ENVLINK) \n')

        infile.write(ds)
       


        string_list = [f'{RAINKEY}{num}{RAINPOST}' for num in dates]
       
        np.save("DateList."+str(dur)+'hr.'+str(process)+".npy", string_list)