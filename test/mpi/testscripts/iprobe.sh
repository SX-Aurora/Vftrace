#!/bin/bash

vftr_binary=iprobe
nprocs=2

export VFTR_SAMPLING="Yes"
export VFTR_MPI_LOG="Yes"
export VFTR_PRECISE="MPI_*|mpi_*"

nb=$(bc <<< "32*${RANDOM}")

${MPI_EXEC} ${MPI_OPTS} ${NP} ${nprocs} ./${vftr_binary} ${nb} || exit 1

irank=1

../../../tools/vftrace_vfd_dump ${vftr_binary}_${irank}.vfd

n=$(../../../tools/vftrace_vfd_dump ${vftr_binary}_${irank}.vfd | \
    grep -i "call MPI_Iprobe" | wc -l)
if [[ ${n} -le 0 ]] ; then
   exit 1;
fi

n=$(../../../tools/vftrace_vfd_dump ${vftr_binary}_${irank}.vfd | \
    grep -i "exit MPI_Iprobe" | wc -l)
if [[ ${n} -le 0 ]] ; then
   exit 1;
fi
