#!/usr/bin/env python3
"""
Annual Maxima Calculation for Watershed Average Precipitation Data

This script calculates annual maxima (AM) precipitation values for specified storm
durations using a moving window approach. It processes watershed area precipitation
(WAP) data and identifies the maximum cumulative precipitation over sliding time
windows of specified duration.

Methodology:
    1. Loads WAP NetCDF files for the specified time period
    2. Applies seasonal filtering if start/end dates are provided
    3. Uses cumulative summation with time-shifted differencing to calculate
       moving window precipitation totals
    4. Identifies maximum values and their temporal occurrence
    5. Tracks file dates spanning each maximum event for use in later codes

Mathematical Approach:
    - Cumulative sum over time: C(t) = Σ(precip[0:t])
    - Time-shifted cumulative: C'(t) = C(t-duration)
    - Moving window sum: MW(t) = C(t) - C'(t)
    - Annual maximum: AM = max(MW(t)) over all t

Usage:
    python 2.AMC.py <duration> <year> <start_mmdd> <end_mmdd> <am_key> <out_key>

Arguments:
    duration    : Storm duration in hours for moving window analysis
    year        : Year to process (YYYY format)
    start_mmdd  : Starting month-day (MMDD) for seasonal filtering
    end_mmdd    : Ending month-day (MMDD) for seasonal filtering  
    am_key      : Prefix for output filename identification
    out_key     : Suffix for output filename identification

Input Requirements:
    - WAP NetCDF4 files (*.nc4) in current directory
    - Files must contain 'WAP' variable and 'time' dimension
    - Time coordinates should be datetime64 format

Output:
    - NetCDF4 file: <am_key>Maximum.<year>.<out_key>.nc4
    - Contains: WAP maxima, time indices, start/end times, file date arrays
    - All arrays compressed with zlib level 9

Dependencies:
    - xarray: NetCDF file handling and array operations
    - numpy: Numerical computations
    - pandas: Date/time manipulation
    - netCDF4: Multi-file dataset handling
    - datetime: Date arithmetic

Author: Benjamin FitzGerald
Date: Last updated 06/02/2025
Version: 2.1
Compatible with: AMC.sh executable and AMC.sub submit file
"""

import xarray as xr 
import sys 
from netCDF4 import MFDataset
import numpy as np
import math
import pandas as pd
from datetime import datetime, timedelta

def get_file_date_str(timestamp: np.datetime64) -> str:
    """
    Convert datetime64 timestamp to file date string format.
    
    Determines which daily file a given hour belongs to, accounting for
    the fact that each daily file contains hours 01:00-00:00 (next day).
    Hour 00:00 belongs to the previous day's file.
    
    Parameters:
    -----------
    timestamp : np.datetime64
        Input timestamp to convert
        
    Returns:
    --------
    str
        Date string in YYYYMMDD format representing the file date
    """
    ts = pd.to_datetime(timestamp)
    if ts.hour == 0:
        # Hour 00:00 belongs to previous day's file
        file_date = ts.normalize() - pd.Timedelta(days=1)
    else:
        # All other hours belong to current day's file
        file_date = ts.normalize()
    return file_date.strftime("%Y%m%d")

def date_range_strings(start_date_str: str, end_date_str: str) -> list:
    """
    Generate list of date strings between two dates (inclusive).
    
    Parameters:
    -----------
    start_date_str : str
        Start date in YYYYMMDD format
    end_date_str : str
        End date in YYYYMMDD format
        
    Returns:
    --------
    list
        List of date strings in YYYYMMDD format
        
    Raises:
    -------
    ValueError
        If end date is before start date
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

# Parse command line arguments
stormDuration = int(sys.argv[1])          # Storm duration in hours
year = sys.argv[2]                        # Year to process (YYYY)
RAINVAR = 'WAP'                          # Precipitation variable name (fixed)
start_date = year + sys.argv[3]          # Start date: YYYYMMDD
end_date = year + sys.argv[4]            # End date: YYYYMMDD
amkey = sys.argv[5]                      # Output filename prefix which comes from type of maximum (e.g. annual, AMJ, etc.) 
outkey = sys.argv[6]                     # Output filename suffix includes precipitation data, duration

# Load all WAP NetCDF4 files in current directory
print(f"Loading WAP data for {year}...")
RAINFILE = xr.open_mfdataset('*.nc4')

# Apply seasonal time range filtering if specific dates are provided
print(f"Filtering data for period: {start_date} to {end_date}")
start_dt = pd.to_datetime(start_date)
end_dt = pd.to_datetime(end_date) + pd.Timedelta(hours=23)  # Include full end date
RAINFILE = RAINFILE.sel(time=slice(start_dt, end_dt))

# Determine number of days needed for file tracking based on storm duration
if int(stormDuration) <= 25:
    nDays = 2  # Short duration storms span at most 2 file days
else:
    # Longer storms may span multiple days
    nDays = math.ceil(int(stormDuration) / 24) + 1

print(f"Processing {stormDuration}-hour storm duration...")

# ANNUAL MAXIMA CALCULATION USING MOVING WINDOW APPROACH
# =====================================================
# Method: Use cumulative summation with time-shifted differencing
# 1. Calculate cumulative sum over time: cumsum(t)
# 2. Create time-shifted version: cumsum(t-duration)  
# 3. Difference gives moving window sum: cumsum(t) - cumsum(t-duration)
# 4. Find maximum of all moving window sums

# Step 1: Calculate cumulative precipitation sum over time
print("Computing cumulative precipitation sums...")
cumulative = RAINFILE.cumsum(dim='time', skipna=False)

# Step 2: Create time-shifted copy of cumulative sums
ccum = cumulative.copy(deep='True')
SCC = ccum.shift(time=stormDuration)  # Shift by storm duration

# Step 3: Calculate moving window sums via differencing
print("Computing moving window precipitation totals...")
Diff = cumulative - SCC  # This gives sum over [t-duration, t] window

# Step 4: Find maximum precipitation over all moving windows
print("Identifying annual maxima...")
WSMax = Diff.max(dim='time').compute()  # Maximum values
# WSMax = WSMax.round()  # Optional rounding (commented out)

# Find time indices where maxima occur (excluding initial period affected by shift)
WIMax = Diff.isel(time=slice(stormDuration, np.size(Diff.time.values))).argmax(dim='time', skipna=False).compute()

# Mask time indices where no valid maximum exists (all NaN precipitation)
WIMax = WIMax.where(WSMax.WAP.notnull())

# Rename variable for clarity
WIMax = WIMax.rename({"WAP": "timeIndex"})

# TEMPORAL METADATA EXTRACTION
# ============================
# Extract file dates and timing information for each maximum event

print("Extracting temporal metadata for maximum events...")

# Get time coordinate array for indexing
TList = RAINFILE["time"].values

# Get time index array and find valid (non-NaN) locations
TIs = WIMax.timeIndex.values
indices = np.where(np.isnan(TIs[:, :]) == False)

# Initialize arrays for storing temporal metadata
dates_array = np.full((nDays, TIs.shape[0], TIs.shape[1]), np.nan)      # File dates for each maximum
am_start_array = np.full((TIs.shape[0], TIs.shape[1]), np.nan)          # Start time of maximum period
am_end_array = np.full((TIs.shape[0], TIs.shape[1]), np.nan)            # End time of maximum period

# Process each valid grid cell with a maximum event
for aa in range(np.shape(indices)[1]):
    # Get time index (add 1 to account for argmax being 0-based after slice)
    TI = TIs[indices[0][aa], indices[1][aa]] + 1
    
    # Extract start and end times of the maximum precipitation period
    am_start_array[indices[0][aa], indices[1][aa]] = TList[int(TI)]
    am_end_array[indices[0][aa], indices[1][aa]] = TList[int(TI) + stormDuration - 1]
    
    # Determine file date range for this maximum event
    start = get_file_date_str(TList[int(TI) - 1])
    end = get_file_date_str(TList[int(TI) - 1 + stormDuration])
    
    # Generate list of all file dates spanning this maximum event
    dates = date_range_strings(start, end)
    
    # Store file dates in array (handle variable length date ranges)
    if np.size(dates) == nDays:    
        dates_array[:, indices[0][aa], indices[1][aa]] = dates
    else:
        # Partial fill for shorter date ranges
        dates_array[0:np.size(dates), indices[0][aa], indices[1][aa]] = dates

# FINAL OUTPUT PREPARATION
# ========================
# Combine all results into comprehensive output dataset

print("Preparing output dataset...")

# Start with maximum precipitation values
AM = WSMax        

# Add temporal metadata arrays
AM['file_days'] = (['nDays', 'latitude', 'longitude'], dates_array)     # File dates for data provenance
AM['WAM_start'] = (['latitude', 'longitude'], am_start_array)           # Maximum period start times
AM['WAM_end'] = (['latitude', 'longitude'], am_end_array)               # Maximum period end times

# Add year dimension for multi-year analysis compatibility
AM = AM.expand_dims(dim={"year": [int(year)]})

# Generate output filename and save with compression
output_filename = f"{amkey}Maximum.{str(year)}.{outkey}.nc4"
print(f"Writing results to: {output_filename}")

AM.to_netcdf(
    output_filename,
    encoding={
        RAINVAR: {"zlib": True, "complevel": 9},          # Compress precipitation data
        'file_days': {"zlib": True, "complevel": 9},      # Compress file date arrays
        'WAM_start': {"zlib": True, "complevel": 9},      # Compress start time array
        'WAM_end': {"zlib": True, "complevel": 9}         # Compress end time array
    },
    format='NETCDF4'
)

print(f"Annual maxima calculation complete for {year}")
print(f"Storm duration: {stormDuration} hours")
print(f"Season: {start_date} to {end_date}")
