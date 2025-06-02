#!/usr/bin/env python3
"""
L-Moments all Longitude Combiner

Usage:
    python 2p.CombineLMs.py <LMLoc> <AMKey> <file_key1> <file_key2>


Arguments:
    LMLOC_key    : Where are LM column files coming from
    AM_key       : identifier used in annual maxima filenames
    file_key1    : Suffix identifier used in LM filenames
    file_key2    : Suffix identifier used in LM filenames

Author: Benjamin FitzGerald
Date: 6/2/2025
Version: 2.1
Note: Companion script to annual maxima calculator for multi-year analysis
"""


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
