#!/bin/bash

CC=mpinfort
ARGS=$@
FILE=`echo $ARGS | awk -F"-c" '{print $2}'`
echo "FILE: $FILE"
set -x
/hpc/nec_scratch/extcweis/Vftrace/tools/compile_wrappers/vftr_fort.py $FILE
cp $FILE $FILE.orig
cp $FILE.vftr $FILE
$CC $ARGS
