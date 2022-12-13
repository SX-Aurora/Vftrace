#!/bin/bash

source ${srcdir}/../environment/filenames.sh

vftr_binary=cregions4
configfile=${vftr_binary}.json
nprocs=1
maxnreg=$(bc <<< "${RANDOM}%5+5")

determine_bin_prefix $vftr_binary

logfile=$(get_logfile_name ${vftr_binary} "all")
vfdfile=$(get_vfdfile_name ${vftr_binary} "0")

# create logfile
echo "{\"sampling\": {\"active\": true}}" > ${configfile}
export VFTR_CONFIG=${configfile}

if [ "x${HAS_MPI}" == "xYES" ]; then
   ${MPI_EXEC} ${MPI_OPTS} ${NP} ${nprocs} ./${vftr_binary} ${maxnreg} || exit 1
else
   ./${vftr_binary} ${maxnreg} || exit 1
fi

cat ${logfile}

for ireg in $(seq 1 1 ${maxnreg});
do
   # build the nested regions stack string to search in log file
   stackstr=""
   for istack in $(seq 1 1 ${ireg});
   do
      stackstr="user-region-${istack}<${stackstr}"
   done

   inprof=$(cat ${logfile} | \
            grep " ${stackstr}" | \
            wc -l)
   if [ "${inprof}" -ne "1" ] ; then
      echo "User region stack \"${stackstr}\" not found in log file the expected amount"
      exit 1;
   fi
done


../../tools/vftrace_vfd_dump ${vfdfile}

for ireg in $(seq 1 1 ${maxnreg});
do

   # build the nested regions stack string to search in log file
   stackstr=""
   for istack in $(seq 1 1 ${ireg});
   do
      stackstr="user-region-${istack}[*]<${stackstr}"
   done

   ncalls=$(../../tools/vftrace_vfd_dump ${vfdfile} | \
            grep "call ${stackstr}" | \
            wc -l)
   if [ "${ncalls}" -ne "1" ] ; then
      echo "Call to user region stack \"${stackstr}\" not found in vfd file"
      exit 1;
   fi
   
   nexits=$(../../tools/vftrace_vfd_dump ${vfdfile} | \
            grep "exit ${stackstr}" | \
            wc -l)
   
   if [ "${nexits}" -ne "1" ] ; then
      echo "Exit from user region \"${stackstr}\" not found in vfd file"
      exit 1;
   fi
done

