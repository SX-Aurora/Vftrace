#!/bin/bash

vftr_binary=neighbor_alltoallv_sync_time
configfile=${vftr_binary}.json
nprocs=4
ntrials=1

echo "{\"logfile_for_ranks\": \"all\", \"mpi\": {\"active\": true, \"show_sync_time\": true}}" > ${configfile}
export VFTR_CONFIG=${configfile}

for itrial in $(seq 1 1 ${ntrials});
do
   # Generate a random message size
   nb=$(bc <<< "32*${RANDOM}")
   ${MPI_EXEC} ${MPI_OPTS} ${NP} ${nprocs} ./${vftr_binary} ${nb} || exit 1

   for irank in $(seq 0 1 $(bc <<< "${nprocs}-1"));
   do  
      cat ${vftr_binary}_${irank}.log
      # Count the logs
      count=$(cat ${vftr_binary}_${irank}.log | \
              grep -i "MPI_Neighbor_alltoallv_sync[ |]*MPI_Neighbor_alltoallv" | \
              wc -l);
      if [[ "${count}" -lt "1" ]] ; then
         echo "Sync region not found on rank ${irank}"
         exit 1;
      fi  
      callcount=$(cat ${vftr_binary}_${irank}.log | \
                  grep -i "MPI_Neighbor_alltoallv_sync[ |]*MPI_Neighbor_alltoallv" | \
                  awk '{print $1}');
      if [[ "${callcount}" -ne "1" ]] ; then
         echo "Mismatch in callcount of sync region."
         echo "expected 1 call, found ${callcount}."
         exit 1;
      fi  
   done
done
