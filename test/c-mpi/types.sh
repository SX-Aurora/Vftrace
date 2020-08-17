#!/bin/bash

vftr_binary=types
nprocs=2

export VFTR_SAMPLING="Yes"
export VFTR_MPI_LOG="Yes"
export VFTR_PRECISE="MPI_*"

mpirun -np ${nprocs} ./${vftr_binary} || exit 1

for ivfd in $(seq 0 1 $(bc <<< "${nprocs} - 1"));
do

   ../../tools/tracedump ${vftr_binary}_${ivfd}.vfd

   itype=0
   for mpitype in MPI_CHAR MPI_SHORT MPI_INT MPI_LONG MPI_LONG_LONG_INT \
                  MPI_UNSIGNED_CHAR MPI_UNSIGNED_SHORT MPI_UNSIGNED \
                  MPI_UNSIGNED_LONG MPI_UNSIGNED_LONG_LONG  \
                  MPI_FLOAT MPI_DOUBLE MPI_LONG_DOUBLE  \
                  MPI_WCHAR MPI_C_BOOL  \
                  MPI_INT8_T MPI_INT16_T MPI_INT32_T MPI_INT64_T  \
                  MPI_UINT8_T MPI_UINT16_T MPI_UINT32_T MPI_UINT64_T  \
                  MPI_C_COMPLEX MPI_C_DOUBLE_COMPLEX MPI_C_LONG_DOUBLE_COMPLEX  \
                  MPI_BYTE ;
   do
      ((itype+=1))
      tmptype=$(../../tools/tracedump ${vftr_binary}_${ivfd}.vfd | \
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
