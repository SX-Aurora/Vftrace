#!/bin/bash
vftr_binary=init_finalize
nprocs=4

export VFTR_SAMPLING="Yes"
export VFTR_MPI_LOG="Yes"
export VFTR_PRECISE="MPI_*|mpi_*"

mpirun -np ${nprocs} ./${vftr_binary} || exit 1

cat ${vftr_binary}_0.log

n=$(cat ${vftr_binary}_0.log | \
    grep -i "MPI_Init<\|MPI_Init_f08" | wc -l)
if [[ ${n} -le 0 ]] ; then
   exit 1;
fi

n=$(cat ${vftr_binary}_0.log | \
    grep -i "MPI_Finalize<\|MPI_Finalize_f08" | wc -l)
if [[ ${n} -le 0 ]] ; then
   exit 1;
fi
