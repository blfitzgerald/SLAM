import os
from pathlib import Path
import sys

# Customize this to the root folder containing year subfolders
root_dir = Path(sys.argv[1])  # <-- CHANGE THIS
output_file = sys.argv[2]          # Name of the output file
files_per_line = int(sys.argv[3])           # Number of file paths per line

# The suffix to append at the end of every line
line_suffix = ",  $(WSLOC)$(WSNAME).shp, $(WSLOC)$(WSNAME).shx, $(PP2WAPCODE), $(ENVLINK)"

# Find all .nc files recursively and sort them
all_files = sorted([str(p) for p in root_dir.rglob("*.nc")])

# Write them to the output file in lines of 50
with open(output_file, "w") as f:
    for i in range(0, len(all_files), files_per_line):
        line_files = all_files[i:i + files_per_line]
        line = ", ".join(line_files) + line_suffix + "\n"
        f.write(line)

print(f"Wrote {len(all_files)} file paths to '{output_file}' in lines of {files_per_line}.")