cd $1

for i in $(seq $2 $3); do
  tar cvfz WAPs.$i.tar.gz $i*.WAP.nc4
done