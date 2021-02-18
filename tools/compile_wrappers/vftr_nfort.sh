#!/bin/bash

CC=mpinfort
ARGS=$@
FILE=`echo $ARGS | grep -oE "[A-Za-z0-9_\-\.\/]+.f90"`
OTHER=`echo $ARGS | grep -vE "[A-Za-z_\-\.]+.f90"`
echo "FILE: $FILE"
set -x
/usr/uhome/aurora/ess/esscw/Vftrace/tools/compile_wrappers/vftr_fort.py $FILE
cp $FILE $FILE.orig
cp $FILE.vftr $FILE
#ARGS2=`sed "s/$FILE/$FILE.vftr/g" <<< $ARGS`
echo "ARGS2: $ARGS2"
$CC $ARGS
