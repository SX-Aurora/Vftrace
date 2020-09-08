#!/bin/bash

vftr_binary=fregions2
nprocs=1

export VFTR_SAMPLING="Yes"
export VFTR_PROF_TRUNCATE="no"
export VFTR_REGIONS_PRECISE="yes"

if [ "x$HAS_MPI" == "xYES" ]; then
   ${MPI_EXEC} ${NP} ${nprocs} ./${vftr_binary} || exit 1
else
   ./${vftr_binary} || exit 1
fi

cat ${vftr_binary}_0.log

inprof=$(cat ${vftr_binary}_0.log | \
         grep "user-region-1" | \
         wc -l)
if [ "${inprof}" -ne "2" ] ; then
   echo "User region \"user-region-1\" not found in log file the expected amount"
   exit 1;
fi


../../tools/tracedump ${vftr_binary}_0.vfd

ncalls=$(../../tools/tracedump ${vftr_binary}_0.vfd | \
         grep "call user-region-1" | \
         wc -l)
if [ "${ncalls}" -ne "1" ] ; then
   echo "Call to user region \"user-region-1\" not found in vfd file"
   exit 1;
fi

nexits=$(../../tools/tracedump ${vftr_binary}_0.vfd | \
         grep "exit user-region-1" | \
         wc -l)

if [ "${nexits}" -ne "1" ] ; then
   echo "Exit from user region \"user-region-1\" not found in vfd file"
   exit 1;
fi

