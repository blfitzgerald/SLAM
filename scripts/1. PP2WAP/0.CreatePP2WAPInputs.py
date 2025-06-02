#!/usr/bin/env python3
"""
PP2WAP Input File Generator

This script searches for daily precipitation files in netcdf format in a directory
and generates an output file containing file paths for daily files grouped into lines with a 
specified number of paths per line. Each line is appended with predefined 
suffix values that include the other files need for the code to run.
Usage:
    python 0.CreatePP2WAPInputs.py <root_directory> <output_file> <files_per_line>

Arguments:
    root_directory  : Root directory to search for .nc files recursively
    output_file     : Name of the output file to write the paths to
    files_per_line  : Number of daily precipitation file paths to include per line

Output Format:
    Each line contains comma-separated file paths followed by additional 
    processing parameters: .shp, .shx files, PP2WAPCODE, and ENVLINK.

Author: Benjamin FitzGerald
Date: 6/2/2025
Version: 1.1
"""

import os
from pathlib import Path
import sys

# Customize this to the root folder containing year subfolders
root_dir = Path(sys.argv[1])

# Name of the output file
output_file = sys.argv[2]          

# Number of file paths per line
files_per_line = int(sys.argv[3])           

# The suffix to append at the end of every line is the other files need to run the PP2WAP Code 
line_suffix = ",  $(WSLOC)$(WSNAME).shp, $(WSLOC)$(WSNAME).shx, $(PP2WAPCODE), $(ENVLINK)"

# Find all .nc files recursively and sort them alphabetically
all_files = sorted([str(p) for p in root_dir.rglob("*.nc")])

# Write them to the output file in lines of specified length
with open(output_file, "w") as f:
    # Process files in chunks of 'files_per_line' size
    for i in range(0, len(all_files), files_per_line):
        # Get the current chunk of files
        line_files = all_files[i:i + files_per_line]
        # Create the line with comma-separated paths plus suffix
        line = ", ".join(line_files) + line_suffix + "\n"
        # Write the line to the output file
        f.write(line)

# Print summary of operation
print(f"Wrote {len(all_files)} file paths to '{output_file}' in lines of {files_per_line}.")
