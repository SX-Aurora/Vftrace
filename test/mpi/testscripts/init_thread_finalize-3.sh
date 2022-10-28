#!/bin/bash

set -x

vftr_binary=init_thread_finalize_3
configfile=${vftr_binary}.json
nprocs=4

echo "{\"logfile_for_ranks\": \"all\", \"sampling\": {\"active\": true}, \"mpi\": {\"active\": true}}" > ${configfile}
export VFTR_CONFIG=${configfile}
export VFTR_LOGFILE_FOR_RANKS="all"

${MPI_EXEC} ${MPI_OPTS} ${NP} ${nprocs} ./${vftr_binary} || exit 1

cat ${vftr_binary}_0.log

n=$(cat ${vftr_binary}_0.log | \
   grep -i "MPI_Init_thread<\|MPI_Init_thread_f08<" | wc -l)
if [[ ${n} -le 0 ]] ; then
   exit 1;
fi

n=$(cat ${vftr_binary}_0.log | \
    grep -i "MPI_Finalize<\|MPI_Finalize_f08<" | wc -l)
if [[ ${n} -le 0 ]] ; then
   exit 1;
fi
