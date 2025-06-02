#!/bin/bash
# have job exit if any command returns with non-zero exit status (aka failure)
set -e

# replace env-name on the right hand side of this line with the name of your conda environment
ENVNAME=$1
# if you need the environment directory to be named something other than the environment name, change this line
ENVDIR=$ENVNAME

# these lines handle setting up the environment; you shouldn't have to modify them
export PATH
mkdir $ENVDIR
tar -xzf $ENVNAME.tar.gz -C $ENVDIR
. $ENVDIR/bin/activate

# modify this line to run your desired Python script and any other work you need to do
python3 $2 $3 $4 $5 $6 $7 $8 $9 ${10} ${11} ${12} ${13} ${14} ${15} ${16} ${17} ${18} ${19}