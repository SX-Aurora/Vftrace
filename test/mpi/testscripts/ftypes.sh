#!/bin/bash

vftr_binary=ftypes
configfile=${vftr_binary}.json
nprocs=2

echo "{\"sampling\": {\"active\": true}}" > ${configfile}
export VFTR_CONFIG=${configfile}

${MPI_EXEC} ${MPI_OPTS} ${NP} ${nprocs} ./${vftr_binary} || exit 1

for ivfd in $(seq 0 1 $(bc <<< "${nprocs} - 1"));
do

   ../../../tools/vftrace_vfd_dump ${vftr_binary}_${ivfd}.vfd

   itype=0
   for mpitype in MPI_INTEGER \
                  MPI_REAL MPI_DOUBLE_PRECISION \
                  MPI_COMPLEX \
                  MPI_LOGICAL \
                  MPI_CHARACTER
   do
      ((itype+=1))
      tmptype=$(../../../tools/vftrace_vfd_dump ${vftr_binary}_${ivfd}.vfd | \
                awk '($2=="send" || $2=="recv") && $3!="end"{getline;print;}' | \
                sed 's/=/ /g;s/(/ /g' | \
                awk '{print $4}' | \
                head -n ${itype} | tail -n 1)

      if [ ! "${mpitype}" = "${tmptype}" ] ; then
         echo "Expected MPI_TYPE ${mpitype} but ${tmptype} was used."
         # Currently NEC-MPI does not destinguish between MPI_CHAR and MPI_CHARACTER
         if [ ! "${mpitype}" = "MPI_CHARACTER" && ! "${tmptype}" = "MPI_CHAR" ] ; then
            exit 1;
         fi
      fi
   
   done
done
