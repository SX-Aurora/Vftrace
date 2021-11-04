#!/bin/bash

vftr_binary=alltoallw_sync_time
nprocs=4
ntrials=1

export VFTR_SAMPLING="No"
export VFTR_MPI_LOG="No"
export VFTR_MPI_SHOW_SYNC_TIME="Yes"
export VFTR_PROF_TRUNCATE="No"
export VFTR_LOGFILE_FOR_RANKS="all"

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
              grep -i "MPI_Alltoallw_sync[ |]*MPI_Alltoallw" | \
              wc -l);
      if [[ "${count}" -lt "1" ]] ; then
         echo "Sync region not found on rank ${irank}"
         exit 1;
      fi  
      callcount=$(cat ${vftr_binary}_${irank}.log | \
                  grep -i "MPI_Alltoallw_sync[ |]*MPI_Alltoallw" | \
                  awk '{print $1}');
      if [[ "${callcount}" -ne "1" ]] ; then
         echo "Mismatch in callcount of sync region."
         echo "expected 1 call, found ${callcount}."
         exit 1;
      fi  
   done
done
