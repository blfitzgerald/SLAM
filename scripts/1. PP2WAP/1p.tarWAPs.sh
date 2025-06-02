# This executable moves the WAP output files in to a subfolder and tars them based on year which is how the should be given to the AMC job submission
mkdir -p $1/WAPs

cd $1
mv -p WAP.*.nc4 WAPs

cd WAPs

for i in $(seq $2 $3); do
  tar cfz WAPs.$i.$4.tar.gz WAP.$i*.$4.nc4
done


