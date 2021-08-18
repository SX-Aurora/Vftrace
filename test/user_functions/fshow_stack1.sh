#!/bin/bash

vftr_binary=fshow_stack1
nprocs=1
export VFTR_LOGFILE_BASENAME=${vftr_binary}
logfile=${vftr_binary}_0.log

if [ "x$HAS_MPI" == "xYES" ]; then
   ${MPI_EXEC} ${MPI_OPTS} ${NP} ${nprocs} ./${vftr_binary} || exit 1
else
   ./${vftr_binary} || exit 1
fi

cat $logfile
echo "***********"

