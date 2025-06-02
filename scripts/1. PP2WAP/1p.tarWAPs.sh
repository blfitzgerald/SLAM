mkdir -p $1/WAPs

cd $1
#mv -p WAP.*.nc4 WAPs

cd WAPs

rm WAPs.*.tar.gz

for i in $(seq $2 $3); do
  tar cfz WAPs.$i.$4.tar.gz WAP.$i*.$4.nc4
done


