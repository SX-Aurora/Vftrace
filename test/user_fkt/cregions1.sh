#!/bin/bash

vftr_binary=cregions1
nprocs=1

if [ "x$HAS_MPI" == "xYES" ]; then
   ${MPI_EXEC} ${NP} ${nprocs} ./${vftr_binary} || exit 1
else
   ./${vftr_binary} || exit 1
fi

cat ${vftr_binary}_0.log

inprof=$(cat ${vftr_binary}_0.log | \
         grep "user-region-1" | \
         wc -l)
if [ "$ncalls" -ne "2" ] ; then
   echo "User region not found in log file the expected amount"
   exit 1;
fi


../../tools/tracedump ${vftr_binary}_0.vfd

ncalls=$(../../tools/tracedump ${vftr_binary}_0.vfd | \
         grep "call user-region-1" | \
         wc -l)
if [ "$ncalls" -ne "1" ] ; then
   echo "Call to user region not found in vfd file"
   exit 1;
fi

nexits=$(../../tools/tracedump ${vftr_binary}_0.vfd | \
         grep "exit user-region-1" | \
         wc -l)

if [ "$ncalls" -ne "1" ] ; then
   echo "Exit from user region not found in vfd file"
   exit 1;
fi

