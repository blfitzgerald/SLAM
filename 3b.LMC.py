## Last updated 11/11/2024
# Title: L-Moment Calculation
# Author: Benjamin FitzGerald
# Purpose: The goal is to calculate the mean of a watershed's average annual maxima, then calculate the L-moments over that area. 
# Inputs: 1) WSArray.npy, 2) Dur, 3) set number (which job from submit file) 4/5) Annual Maxima files, 6) DateList.npy 7)PointList.np, 8) Precip Data for days in DateList
# Inputs from code are compatible with execudable, LMC.sh, and submit file, LMC.sub though the sys.argv inputs could be replaced by the user. 
# Outputs: LM.$setNum.nc

import xarray as xr
import numpy as np
import lmoments3 as lm3
import sys
import glob
import math

WSArray=np.load("WSArray.npy")

dur=sys.argv[2]
setNum=str(sys.argv[1])

AMwhen=xr.open_dataset(str(dur)+"hr.AMtimeIndex.nc4")
AM=xr.open_dataset(str(dur)+"hr.fullAM.nc4")

AMwhen['year'] = AMwhen['year'].astype(int)
AM['year'] = AM['year'].astype(int)

WSArray[WSArray>0]=1
WSShape=np.shape(WSArray)

date_pattern='DateList.*.npy'
point_pattern='PointList.*.npy'

date_file = glob.glob(date_pattern)
point_file=glob.glob(point_pattern)

pointSet=np.load(point_file[0])
DateList=np.load(date_file[0])

AMFilesFull=xr.open_mfdataset(DateList)
AMFilesFull
latitude=AMFilesFull.latitude.values
longitude=AMFilesFull.longitude.values

if int(dur)<=25:
    nDays=2
else:
    nDays=math.ceil(int(dur)/24)+1

ydim=np.size(latitude)
xdim=np.size(longitude)
    
l1=np.zeros([ydim,xdim])
l2=np.zeros([ydim,xdim])
l3=np.zeros([ydim,xdim])
l4=np.zeros([ydim,xdim])
lcv=np.zeros([ydim,xdim])

lmomsDS=xr.Dataset({'l1':(['latitude','longitude'], l1),
			'l2':(['latitude','longitude'], l2),
			'l3':(['latitude','longitude'], l3),
			'l4':(['latitude','longitude'], l4),
			'lcv':(['latitude','longitude'], lcv)},
			coords={'latitude':latitude, 'longitude':longitude})

yearlist=AMwhen.year.values
year1=yearlist[0]       
for aa in range(np.shape(pointSet)[0]):
	lat=pointSet[aa,0]
	lon=pointSet[aa,1]
	print(lat)
	print(lon)
	for yy in range(np.size(yearlist)): 
		year=yy+year1
		times=np.load(str(year)+".times.npy")
		timeIndices=AMwhen.timeIndex.sel(year=year, longitude=lon, latitude=lat, method='nearest').values 
            
		hourLast=times[int(timeIndices)]
		hour=times[int(timeIndices)]

		for bb in range(1, nDays):
			hour=hour-np.timedelta64(24, 'h')
        
		yy1=np.where(latitude==lat)[0][0]
		xx1=np.where(longitude==lon)[0][0]
		
		lat2=latitude[yy1-WSShape[0]+1]
		lon2=longitude[xx1+WSShape[1]-1]
		print(hour)
		print(hour+np.timedelta64(1, 'h'))
		print(hourLast)
		AMFiles=AMFilesFull.sel(time=slice(hour+np.timedelta64(1, 'h'),hourLast), latitude=slice(lat2,lat),longitude=slice(lon,lon2)).load()
		dayMasked=np.flip(WSArray,axis=0)*AMFiles
		#dayMasked=WSArray*AMFiles

		if yy==0:
			WSSum=dayMasked.sum(dim=['time'])
          
		if yy>0:
			WSSumNew=dayMasked.sum(dim=['time'])
			WSSum=WSSum+WSSumNew                                                   #Lei: accumulated summed WS AM
           
	PointsTotal=WSSum.precrate.where(WSSum.precrate>0).values
	flattened_list = PointsTotal.flatten()
	filtered_list = flattened_list[~np.isnan(flattened_list)].tolist()
	filtered_list = np.array(filtered_list) / (np.size(yearlist))                          #Lei: annual average AM

	lmoms=lm3.lmom_ratios(filtered_list,4)

	lmomsDS['l1'].loc[latitude==lat, longitude==lon]=lmoms[0]
	lmomsDS['l2'].loc[latitude==lat, longitude==lon]=lmoms[1]
	lmomsDS['l3'].loc[latitude==lat, longitude==lon]=lmoms[2]
	lmomsDS['l4'].loc[latitude==lat, longitude==lon]=lmoms[3]
	lmomsDS['lcv'].loc[latitude==lat, longitude==lon]=lmoms[1]/lmoms[0]

lmomsDS=lmomsDS.where(lmomsDS.l1>0)
comp = dict(zlib=True, complevel=5)
encoding = {var: comp for var in lmomsDS.data_vars}

lmomsDS.to_netcdf("LM."+str(dur)+'hr.'+setNum+".nc", format='netcdf4', engine='netcdf4', encoding=encoding)        
