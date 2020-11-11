#!/bin/bash

vftr_binary=derived_ftypes
nprocs=2

export VFTR_SAMPLING="Yes"
export VFTR_MPI_LOG="Yes"
export VFTR_PRECISE="MPI_*|mpi_*"

${MPI_EXEC} ${MPI_OPTS} ${NP} ${nprocs} ./${vftr_binary} || exit 1

mpitype=MPI_DERIVED_TYPE

for ivfd in $(seq 0 1 $(bc <<< "${nprocs}-1"));
do

   ../../../tools/tracedump ${vftr_binary}_${ivfd}.vfd

   tmptype=$(../../../tools/tracedump ${vftr_binary}_${ivfd}.vfd | \
             awk '($2=="send" || $2=="recv") && $3!="end"{getline;print;}' | \
             sed 's/=/ /g;s/(/ /g' | \
             awk '{print $4}' | \
             head -n 1)

   if [ ! "${mpitype}" = "${tmptype}" ] ; then
      echo "Expected MPI_TYPE ${mpitype} but ${tmptype} was used."
      exit 1;
   fi
done
