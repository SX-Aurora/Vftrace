#!/bin/bash

vftr_binary=ftypes
nprocs=2

export VFTR_SAMPLING="Yes"
export VFTR_MPI_LOG="Yes"
export VFTR_PRECISE="MPI_*|mpi_*"

mpirun -np ${nprocs} ./${vftr_binary} || exit 1

for ivfd in $(seq 0 1 $(bc <<< "${nprocs} - 1"));
do

   ../../../tools/tracedump ${vftr_binary}_${ivfd}.vfd

   itype=0
   for mpitype in MPI_INTEGER \
                  MPI_REAL MPI_DOUBLE_PRECISION \
                  MPI_COMPLEX \
                  MPI_LOGICAL \
                  MPI_CHARACTER 
   do
      ((itype+=1))
      tmptype=$(../../../tools/tracedump ${vftr_binary}_${ivfd}.vfd | \
                awk '($2=="send" || $2=="recv") && $3!="end"{getline;print;}' | \
                sed 's/=/ /g;s/(/ /g' | \
                awk '{print $4}' | \
                head -n ${itype} | tail -n 1)

      if [ ! "${mpitype}" = "${tmptype}" ] ; then
         echo "Expected MPI_TYPE ${mpitype} but ${tmptype} was used."
         exit 1;
      fi
   
   done
done
