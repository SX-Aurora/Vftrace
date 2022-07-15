#!/bin/bash

vftr_binary=exscan_nolog
nprocs=4
ntrials=1

export VFTR_SAMPLING="Yes"
export VFTR_MPI_LOG="No"

for itrial in $(seq 1 1 ${ntrials});
do
   # Generate a random message size
   nb=$(bc <<< "32*${RANDOM}")
   ${MPI_EXEC} ${MPI_OPTS} ${NP} ${nprocs} ./${vftr_binary} ${nb} || exit 1

   for irank in $(seq 0 1 $(bc <<< "${nprocs}-1"));
   do  
      ../../../tools/vftrace_vfd_dump ${vftr_binary}_${irank}.vfd
      # Count the logs
      count=$(../../../tools/vftrace_vfd_dump ${vftr_binary}_${irank}.vfd | \
              awk '($2=="recv" || $2=="send") && $3!="end"{print;}' | \
              wc -l);
      if [[ "${count}" -ne "0" ]] ; then
         echo "Message logging found on rank ${irank}, although it should be disabled!"
         exit 1;
      fi  
   done
done
