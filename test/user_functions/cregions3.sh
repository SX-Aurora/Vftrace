#!/bin/bash

source ${srcdir}/../environment/filenames.sh

vftr_binary=cregions3
configfile=${vftr_binary}.json
nprocs=1
maxnreg=$(bc <<< "${RANDOM}%5+5")

determine_bin_prefix $vftr_binary

logfile=$(get_logfile_name ${vftr_binary} "all")
vfdfile=$(get_vfdfile_name ${vftr_binary} "0")

# create logfile
echo "{\"sampling\": {\"active\": true}, \"papi\": {\"show_tables\": false}}" > ${configfile}
export VFTR_CONFIG=${configfile}

if [ "x${HAS_MPI}" == "xYES" ]; then
   ${MPI_EXEC} ${MPI_OPTS} ${NP} ${nprocs} ./${vftr_binary} ${maxnreg} || exit 1
else
   ./${vftr_binary} ${maxnreg} || exit 1
fi

cat ${logfile}

for ireg in $(seq 1 1 ${maxnreg});
do
   inprof=$(cat ${logfile} | \
            grep "user-region-${ireg}" | \
            wc -l)
   if [ "${inprof}" -ne "2" ] ; then
      echo "User region \"user-region-${ireg}\" not found in $logfile the expected amount. Found $inprof, expected 2"
      exit 1;
   fi
done


../../tools/vftrace_vfd_dump ${vfdfile}

for ireg in $(seq 1 1 ${maxnreg});
do
   ncalls=$(../../tools/vftrace_vfd_dump ${vfdfile} | \
            grep "call user-region-${ireg}" | \
            wc -l)
   if [ "${ncalls}" -ne "1" ] ; then
      echo "Call to user region \"user-region-${ireg}\" not found in vfd file"
      exit 1;
   fi
   
   nexits=$(../../tools/vftrace_vfd_dump ${vfdfile} | \
            grep "exit user-region-${ireg}" | \
            wc -l)
   
   if [ "${nexits}" -ne "1" ] ; then
      echo "Exit from user region \"user-region-${ireg}\" not found in vfd file"
      exit 1;
   fi
done

