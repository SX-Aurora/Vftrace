#!/bin/bash

vftr_binary=fregions3
nprocs=1
maxnreg=$(bc <<< "${RANDOM}%5+5")

export VFTR_SAMPLING="Yes"
export VFTR_PROF_TRUNCATE="no"
export VFTR_REGIONS_PRECISE="yes"
export VFTR_LOGFILE_BASENAME=$vftr_binary

if [ "x$HAS_MPI" == "xYES" ]; then
   ${MPI_EXEC} ${MPI_OPTS} ${NP} ${nprocs} ./${vftr_binary} ${maxnreg} || exit 1
else
   ./${vftr_binary} ${maxnreg} || exit 1
fi

cat ${vftr_binary}_0.log

for ireg in $(seq 1 1 ${maxnreg});
do
   inprof=$(cat ${vftr_binary}_0.log | \
            grep "user-region-${ireg}" | \
            wc -l)
   if [ "${inprof}" -ne "2" ] ; then
      echo "User region \"user-region-${ireg}\" not found in log file the expected amount"
      exit 1;
   fi
done


../../tools/vftrace_vfd_dump ${vftr_binary}_0.vfd

for ireg in $(seq 1 1 ${maxnreg});
do
   ncalls=$(../../tools/vftrace_vfd_dump ${vftr_binary}_0.vfd | \
            grep "call user-region-${ireg}" | \
            wc -l)
   if [ "${ncalls}" -ne "1" ] ; then
      echo "Call to user region \"user-region-${ireg}\" not found in vfd file"
      exit 1;
   fi
   
   nexits=$(../../tools/vftrace_vfd_dump ${vftr_binary}_0.vfd | \
            grep "exit user-region-${ireg}" | \
            wc -l)
   
   if [ "${nexits}" -ne "1" ] ; then
      echo "Exit from user region \"user-region-${ireg}\" not found in vfd file"
      exit 1;
   fi
done

