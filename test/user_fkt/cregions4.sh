#!/bin/bash

vftr_binary=cregions4
nprocs=1
maxnreg=$(bc <<< "${RANDOM}%5+5")

export VFTR_PROF_TRUNCATE="no"

if [ "x$HAS_MPI" == "xYES" ]; then
   ${MPI_EXEC} ${NP} ${nprocs} ./${vftr_binary} ${maxnreg} || exit 1
else
   ./${vftr_binary} ${maxnreg} || exit 1
fi

cat ${vftr_binary}_0.log

for ireg in $(seq 1 1 ${maxnreg});
do
   # build the nested regions stack string to search in log file
   stackstr=""
   for istack in $(seq 1 1 ${ireg});
   do
      stackstr="user-region-${istack}<${stackstr}"
   done

   inprof=$(cat ${vftr_binary}_0.log | \
            grep " ${stackstr}" | \
            wc -l)
   if [ "${inprof}" -ne "1" ] ; then
      echo "User region stack \"${stackstr}\" not found in log file the expected amount"
      exit 1;
   fi
done


../../tools/tracedump ${vftr_binary}_0.vfd

for ireg in $(seq 1 1 ${maxnreg});
do

   # build the nested regions stack string to search in log file
   stackstr=""
   for istack in $(seq 1 1 ${ireg});
   do
      stackstr="user-region-${istack}[*]<${stackstr}"
   done

   ncalls=$(../../tools/tracedump ${vftr_binary}_0.vfd | \
            grep "call ${stackstr}" | \
            wc -l)
   if [ "${ncalls}" -ne "1" ] ; then
      echo "Call to user region stack \"${stackstr}\" not found in vfd file"
      exit 1;
   fi
   
   nexits=$(../../tools/tracedump ${vftr_binary}_0.vfd | \
            grep "exit ${stackstr}" | \
            wc -l)
   
   if [ "${nexits}" -ne "1" ] ; then
      echo "Exit from user region \"${stackstr}\" not found in vfd file"
      exit 1;
   fi
done

