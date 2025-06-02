#!/usr/bin/env python3
"""
Annual Maxima Multi-Year Combiner

Usage:
    python 2p.CombineAM.py <amc_key> <file_key>

Arguments:
    amc_key     : Prefix identifier used in annual maxima filenames
    file_key    : Suffix identifier used in annual maxima filenames
    
Author: Benjamin FitzGerald
Date: 6/2/2025
Version: 2.1
Note: Companion script to annual maxima calculator for multi-year analysis
"""


import xarray as xr
import sys

AMCKEY=sys.argv[1]
FILEKEY=sys.argv[2]

ds=xr.open_mfdataset(AMCKEY+"Maximum.*."+FILEKEY+".nc4")

ds.to_netcdf(AMCKEY+"Maxima."+FILEKEY+".nc4",encoding={"WAP":{"zlib": True, "complevel": 9}, 'file_days':{"zlib": True, "complevel": 9}, 'WAM_start':{"zlib": True, "complevel": 9}, 'WAM_end':{"zlib": True, "complevel": 9}},format='NETCDF4')

