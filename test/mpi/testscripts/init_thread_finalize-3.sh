#!/bin/bash

set -x

vftr_binary=init_thread_finalize_3
nprocs=4

export VFTR_SAMPLING="Yes"
export VFTR_MPI_LOG="Yes"
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