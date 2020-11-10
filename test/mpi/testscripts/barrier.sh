#!/bin/bash

vftr_binary=barrier
nprocs=4

export VFTR_SAMPLING="Yes"
export VFTR_MPI_LOG="Yes"
export VFTR_PRECISE="MPI_*|mpi_*"

${MPI_EXEC} ${MPI_OPTS} ${NP} ${nprocs} ./${vftr_binary} || exit 1

for irank in $(seq 0 1 $(bc <<< "${nprocs}-1"));
do
   ../../../tools/tracedump ${vftr_binary}_${irank}.vfd

   n=$(../../../tools/tracedump ${vftr_binary}_${irank}.vfd | \
       grep -i "call MPI_Barrier" | wc -l)
   if [[ ${n} -le 0 ]] ; then
      exit 1;
   fi

   n=$(../../../tools/tracedump ${vftr_binary}_${irank}.vfd | \
       grep -i "exit MPI_Barrier" | wc -l)
   if [[ ${n} -le 0 ]] ; then
      exit 1;
   fi
done
