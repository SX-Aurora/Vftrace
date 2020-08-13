#!/bin/bash

vftr_binary=init_finalize
nprocs=2

export VFTR_SAMPLING="Yes"
export VFTR_MPI_LOG="Yes"
export VFTR_PRECISE="MPI_*"

mpirun -np ${nprocs} ./${vftr_binary} || exit 1

for irank in $(seq 0 1 $(bc <<< "${nprocs}-1"));
do
   ../../tools/tracedump ${vftr_binary}_${irank}.vfd

   n=$(../../tools/tracedump ${vftr_binary}_${irank}.vfd | \
       grep "exit MPI_Init" | wc -l)
   if [[ ${n} -le 0 ]] ; then
      exit 1;
   fi

   n=$(../../tools/tracedump ${vftr_binary}_${irank}.vfd | \
       grep "call MPI_Finalize" | wc -l)
   if [[ ${n} -le 0 ]] ; then
      exit 1;
   fi
done
