#!/bin/bash

vftr_binary=init_thread_finalize_4
configfile=${vftr_binary}.json
nprocs=4

echo "{\"sampling\": {\"active\": true}}" > ${configfile}
export VFTR_CONFIG=${configfile}

${MPI_EXEC} ${MPI_OPTS} ${NP} ${nprocs} ./${vftr_binary} || exit 1

for irank in $(seq 0 1 $(bc <<< "${nprocs}-1"));
do
   ../../../tools/vftrace_vfd_dump ${vftr_binary}_${irank}.vfd

   n=$(../../../tools/vftrace_vfd_dump ${vftr_binary}_${irank}.vfd | \
       grep -i "exit MPI_Init_thread" | wc -l)
   if [[ ${n} -le 0 ]] ; then
      exit 1;
   fi

   n=$(../../../tools/vftrace_vfd_dump ${vftr_binary}_${irank}.vfd | \
       grep -i "call MPI_Finalize" | wc -l)
   if [[ ${n} -le 0 ]] ; then
      exit 1;
   fi
done
