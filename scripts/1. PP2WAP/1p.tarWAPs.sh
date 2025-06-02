# ============================================
# Script Name:    1p.tarWAPs.sh
# Author:         Benjamin FitzGerald
# Date:           2025-06-02
# Description:    This executable moves the WAP output files in to a subfolder and tars them based on year which is how the should be given to the AMC job submissions
# Usage:          ./1p.tarWAPs.sh <job directory> <year1> <year2> <File Key>
# ============================================

cd $1
mv -p WAP.*.nc4 WAPs

cd WAPs

for i in $(seq $2 $3); do
  tar cfz WAPs.$i.$4.tar.gz WAP.$i*.$4.nc4
done


