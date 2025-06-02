#!/usr/bin/env python3
"""
L-Moment Calculation Setup and Job Partitioning

This script prepares input files for parallel L-moment calculations by partitioning
the computational domain into longitude-based groups. It analyzes annual maxima
precipitation data to identify required precipitation files for each group and
generates job input files for distributed processing.

Purpose:
    - Partition L-moment calculations for parallel execution
    - Create longitude-based computational groups to balance workload
    - Generate input file lists containing all precipitation data needed
    - Prepare job submission files for HPC cluster processing

Methodology:
    1. Load annual maxima dataset to identify valid precipitation locations
    2. Group longitude indices into manageable chunks for parallel processing
    3. Extract unique precipitation file dates required for each longitude group
    4. Generate formatted input strings with file paths and processing parameters
    5. Create job input files compatible with cluster submission systems

Parallel Processing Strategy:
    - Splits domain into longitude bands (default: 15 longitudes per group)
    - Each group processes independently to avoid memory/computational bottlenecks
    - Groups are sized to balance computational load and file I/O requirements
    - Enables efficient scaling across multiple compute nodes

Usage:
    python 3a.LMS.py <data_path> <am_key> <rain_key>

Arguments:
    data_path   : Root directory containing precipitation and annual maxima data
    am_key      : Identifier key for annual maxima files
    rain_key    : Identifier key for precipitation data files

Input Data Requirements:
    - Annual maxima NetCDF4 file: <data_path>/AMs/<am_key>Maxima.<rain_key>.nc4
    - Must contain: WAP variable, file_days array, year/lat/lon coordinates
    - Precipitation files organized by year in subdirectories

Output Files:
    - LMCInput.<rain_key>.<am_key>.txt: Job input file containing file lists
      for each longitude group, formatted for cluster job submission

File Format (per line in output):
    Comma-separated list of precipitation files + processing parameters:
    file1, file2, ..., fileN, LMC_code, data_files, shapefile_paths, environment

Processing Parameters (automatically added):
    - $(PRECIPPATH): Path to precipitation data directory
    - $(PRECIPPREFFIX): Prefix for precipitation filenames  
    - $(PRECIPSUFFIX): Suffix for precipitation filenames
    - $(CODELOC): Location of L-moment calculation code
    - $(LMCCODE): L-moment calculation executable name
    - $(DPLoc): Data processing location
    - $(WSLOC): Watershed shapefile location
    - $(AMLOC): Annual maxima file location
    - $(ENVLINK): Environment setup commands

Dependencies:
    - xarray: NetCDF file handling and data analysis
    - numpy: Array operations and data manipulation
    - numba: JIT compilation for performance (imported but not used in current version)
    - re: Regular expression operations for string formatting
    - sys: Command-line argument parsing

Author: Benjamin FitzGerald
Date: Last updated 6/2/2025
Version: 2.1
"""

import xarray as xr
import numpy as np
from numba import njit, prange
import math
import sys
import re

# Parse command line arguments
path = sys.argv[1]        # Root data directory path
AMKEY = sys.argv[2]       # Annual maxima identifier key
RAINKEY = sys.argv[3]     # Precipitation data identifier key

# Configuration parameters
group_size = 15           # Number of longitude indices per processing group
                         # Adjust based on memory constraints and computational resources

print(f"Setting up L-moment calculation jobs for {AMKEY} precipitation data...")

# Load annual maxima dataset containing precipitation maxima and temporal metadata
am_file_path = path + "AMs//" + AMKEY + "Maxima." + RAINKEY + ".nc4"
print(f"Loading annual maxima data from: {am_file_path}")
AM = xr.open_dataset(am_file_path)

# Extract coordinate arrays and temporal information
latitudes = AM.latitude.values           # Latitude coordinates
longitudes = AM.longitude.values         # Longitude coordinates  
yearlist = AM.year.values               # Years with annual maxima data
yearf = int(yearlist[0])                # First year in dataset
nYears = np.size(AM.year.values)        # Total number of years
yearl = int(yearlist[nYears-1])         # Last year in dataset
nLats = np.size(latitudes)              # Number of latitude points
nLons = np.size(longitudes)             # Number of longitude points

print(f"Dataset spans {nYears} years: {yearf} to {yearl}")
print(f"Spatial domain: {nLats} latitudes × {nLons} longitudes")

# Identify valid precipitation points (where annual maxima > 0)
amwap_array = AM.WAP.values             # Annual maxima precipitation values
indices = np.where(amwap_array > 0)     # Find non-zero precipitation locations

# Extract coordinates of valid precipitation points
years = indices[0]                      # Year indices with valid data
lats = indices[1]                       # Latitude indices with valid data  
lons = indices[2]                       # Longitude indices with valid data

# CREATE LONGITUDE-BASED PROCESSING GROUPS
# ========================================
# Group longitude indices to balance computational load across parallel jobs

# Get unique longitude indices that contain valid precipitation data
lonlist = np.unique(np.sort(lons))
print(f"Processing {len(lonlist)} longitude bands with valid precipitation data")

# Partition longitude indices into groups of specified size
sorted_groups = [np.sort(lonlist[i:i+group_size]) for i in range(0, len(lonlist), group_size)]
num_groups = len(sorted_groups)

print(f"Created {num_groups} processing groups with {group_size} longitudes each")

# Extract file date information for determining required precipitation files
all_dates = AM.file_days.values         # Array containing file dates for each maximum event

# GENERATE JOB INPUT FILE
# =======================
# Create input file containing precipitation file lists for each processing group

output_file_path = path + "LMCInput." + RAINKEY + "." + AMKEY + '.txt'
print(f"Generating job input file: {output_file_path}")

with open(output_file_path, 'w') as infile: 
    
    # Process each longitude group independently
    for aa in range(num_groups):
        print(f"Processing group {aa+1}/{num_groups}...")
        
        # Get longitude range for current group
        lon1 = sorted_groups[aa][0]         # First longitude index in group
        lon2 = sorted_groups[aa][-1]        # Last longitude index in group
        
        # Extract file dates needed for this longitude range
        # Include all dates where any point in this longitude band has annual maxima
        lon_dates = all_dates[:, :, :, lon1:lon2+1]    # Subset dates array
        lon_dates = lon_dates.flatten()                 # Flatten to 1D array
        lon_dates = lon_dates[np.isnan(lon_dates) == False]  # Remove NaN values
        
        # Get unique dates to avoid duplicate file processing
        uniqueDates = np.unique(lon_dates)
        dates = uniqueDates.astype(int)     # Convert to integer format (YYYYMMDD)
        
        # Only process groups that have valid dates
        if np.size(dates) > 0:
            
            # HANDLE LARGE DATE LISTS
            # ======================
            # For groups with many dates, split into chunks to avoid memory issues
            
            if np.size(dates) > 1000:
                # Process in chunks of 1000 dates
                ds = str(dates[0:1000]).replace('\n', '')
                counted = 1000
                remaining = np.size(dates) - counted
                
                # Continue processing remaining dates in chunks
                while remaining > 0:
                    if remaining >= 1000:
                        ds = ds + str(dates[counted:counted+1000]).replace('\n', '')
                        counted = counted + 1000
                        remaining = np.size(dates) - counted
                    else: 
                        ds = ds + str(dates[counted:counted+remaining]).replace('\n', '')
                        counted = counted + remaining
                        remaining = np.size(dates) - counted
            else:
                # Handle smaller date lists in single operation
                ds = str(dates).replace('\n', '')
            
            # FORMAT FILE PATHS FOR CLUSTER PROCESSING
            # =======================================
            # Convert date array to formatted file path string with cluster variables
            
            # Clean up array formatting
            ds = re.sub(r'\]\[', ' ', ds)       # Remove array boundaries between chunks
            
            # Add precipitation file path template
            ds = ds.replace('[', '[$(PRECIPPATH)/YEAR/$(PRECIPPREFFIX)')
            ds = ds.replace(' ', '$(PRECIPSUFFIX), $(PRECIPPATH)/YEAR/$(PRECIPPREFFIX)')
            
            # Substitute actual years into path templates
            for yy in range(yearf, yearl+1):
                year = str(int(yy))
                ds = ds.replace('/YEAR/$(PRECIPPREFFIX)'+year, '/'+year+'/$(PRECIPPREFFIX)'+year)
            
            # Clean up and add processing parameters
            ds = ds.replace('[', '')            # Remove opening bracket
            ds = ds.replace(']', '$(PRECIPSUFFIX), $(CODELOC)/$(LMCCODE), $(DPLoc)/$(DPFile), $(WSLOC)/$(WSNAME).shp, $(WSLOC)/$(WSNAME).shx, $(AMLOC)/$(AMFILE), $(ENVLINK) \n')
            
            # Write formatted line to input file
            infile.write(ds)
            
            print(f"  Group {aa+1}: {np.size(dates)} unique precipitation files required")

print(f"L-moment calculation setup complete!")
print(f"Generated {num_groups} processing groups")
print(f"Input file created: {output_file_path}")
print("Ready for cluster job submission using LMS.sh and LMS.sub")
