#!/bin/bash

source ${srcdir}/../environment/filenames.sh

set -x

vftr_binary=fregions1
nprocs=1

logfile=$(get_logfile_name ${vftr_binary} "all")
vfdfile=$(get_vfdfile_name ${vftr_binary} "0")

export VFTR_SAMPLING="Yes"
export VFTR_REGIONS_PRECISE="yes"

if [ "x${HAS_MPI}" == "xYES" ]; then
   ${MPI_EXEC} ${MPI_OPTS} ${NP} ${nprocs} ./${vftr_binary} || exit 1
else
   ./${vftr_binary} || exit 1
fi

cat ${logfile}

inprof=$(cat ${logfile} | \
         grep "user-region-1" | \
         wc -l)
if [ "${inprof}" -ne "2" ] ; then
   echo "User region \"user-region-1\" not found in log file the expected amount"
   exit 1;
fi


../../tools/vftrace_vfd_dump ${vfdfile}

ncalls=$(../../tools/vftrace_vfd_dump ${vfdfile} | \
         grep "call user-region-1" | \
         wc -l)
if [ "${ncalls}" -ne "1" ] ; then
   echo "Call to user region \"user-region-1\" not found in vfd file"
   exit 1;
fi

nexits=$(../../tools/vftrace_vfd_dump ${vfdfile} | \
         grep "exit user-region-1" | \
         wc -l)

if [ "${nexits}" -ne "1" ] ; then
   echo "Exit from user region \"user-region-1\" not found in vfd file"
   exit 1;
fi

