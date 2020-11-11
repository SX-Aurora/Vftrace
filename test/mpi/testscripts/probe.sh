#!/bin/bash

vftr_binary=probe
nprocs=2

export VFTR_SAMPLING="Yes"
export VFTR_MPI_LOG="Yes"
export VFTR_PRECISE="MPI_*|mpi_*"

nb=$(bc <<< "32*${RANDOM}")

${MPI_EXEC} ${MPI_OPTS} ${NP} ${nprocs} ./${vftr_binary} ${nb} || exit 1

irank=1

../../../tools/tracedump ${vftr_binary}_${irank}.vfd

n=$(../../../tools/tracedump ${vftr_binary}_${irank}.vfd | \
    grep -i "call MPI_Probe" | wc -l)
if [[ ${n} -le 0 ]] ; then
   exit 1;
fi

n=$(../../../tools/tracedump ${vftr_binary}_${irank}.vfd | \
    grep -i "exit MPI_Probe" | wc -l)
if [[ ${n} -le 0 ]] ; then
   exit 1;
fi
