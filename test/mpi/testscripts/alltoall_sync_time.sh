#!/bin/bash

source ${srcdir}/../../environment/filenames.sh

vftr_binary=alltoall_sync_time
configfile=${vftr_binary}.json
nprocs=4
ntrials=1

determine_bin_prefix $vftr_binary

echo "{\"logfile_for_ranks\": \"all\", \"mpi\": {\"active\": true, \"show_sync_time\": true}}" > ${configfile}
export VFTR_CONFIG=${configfile}

for itrial in $(seq 1 1 ${ntrials});
do
   # Generate a random message size
   nb=$(bc <<< "32*${RANDOM}")
   ${MPI_EXEC} ${MPI_OPTS} ${NP} ${nprocs} ./${vftr_binary} ${nb} || exit 1

   for irank in $(seq 0 1 $(bc <<< "${nprocs}-1"));
   do  
      logfile=$(get_logfile_name ${vftr_binary} ${irank})
      cat ${logfile}
      # Count the logs
      count=$(cat ${logfile} | \
              grep -i "MPI_Alltoall_sync[ |]*MPI_Alltoall" | \
              wc -l);
      if [[ "${count}" -lt "1" ]] ; then
         echo "Sync region not found on rank ${irank}"
         exit 1;
      fi  
      callcount=$(cat ${logfile} | \
                  grep -i "MPI_Alltoall_sync[ |]*MPI_Alltoall" | \
                  awk '{print $1}');
      if [[ "${callcount}" -ne "1" ]] ; then
         echo "Mismatch in callcount of sync region."
         echo "expected 1 call, found ${callcount}."
         exit 1;
      fi  
   done
done
