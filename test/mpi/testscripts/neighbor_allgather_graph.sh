#!/bin/bash

vftr_binary=neighbor_allgather_graph
nprocs=4
ntrials=1

export VFTR_SAMPLING="Yes"
export VFTR_MPI_LOG="Yes"
export VFTR_PRECISE="MPI_*|mpi_*"

for itrial in $(seq 1 1 ${ntrials});
do
   # Generate a random message size
   nb=$(bc <<< "32*${RANDOM}")
   ${MPI_EXEC} ${MPI_OPTS} ${NP} ${nprocs} ./${vftr_binary} ${nb} || exit 1

   # A neighbor list needs to be created for checking the communication
   neighborlist[0]="1 2"
   neighborlist[1]="0 2"
   neighborlist[2]="0 1 3"
   neighborlist[3]="2 3"
   msgnum[0]="2 1"
   msgnum[1]="2 1"
   msgnum[2]="0 1 3"
   msgnum[3]="2 3"
   # check each rank for the correct message communication
   # patterns in the vfd file
   for irank in $(seq 0 1 $(bc <<< "${nprocs}-1"));
   do
      ../../../tools/vftrace_vfd_dump ${vftr_binary}_${irank}.vfd

      npeers=$(echo "${neighborlist[${irank}]}" | wc -w)
      for peer_idx in $(seq 1 1 ${npeers});
      do
         jrank=$(echo "${neighborlist[${irank}]}" | awk -v i=${peer_idx} '{print $i}')
         msgnum_val=$(echo "${msgnum[${irank}]}" | awk -v i=${peer_idx} '{print $i}')
         ipeer=$(bc <<< "${jrank} + 1")
         # Validate sending
         # check if every message appears the correct amount of times
         msgcount=$(../../../tools/vftrace_vfd_dump ${vftr_binary}_${irank}.vfd | \
                    awk '$2=="send" && $3!="end"{getline;print;}' | \
                    grep "peer=${jrank}" | \
                    wc -l)
         if [[ "${msgcount}" -ne "${msgnum_val}" ]] ; then
            echo "Message send from rank ${irank} to ${jrank} appeared ${msgcount} times!"
            echo "Was expecting message to appear ${msgnum_val} times!"
            exit 1;
         fi
         # Get actually used message size
         count=$(../../../tools/vftrace_vfd_dump ${vftr_binary}_${irank}.vfd | \
                 awk '$2=="send" && $3!="end"{getline;print;}' | \
                 grep "peer=${jrank}" | \
                 sed 's/=/ /g' | \
                 awk '{print $2}' | \
                 head -n ${ipeer} | tail -n 1)
         # get peer process
         peer=$(../../../tools/vftrace_vfd_dump ${vftr_binary}_${irank}.vfd | \
                awk '$2=="send" && $3!="end"{getline;print;}' | \
                grep "peer=${jrank}" | \
                sed 's/=/ /g' | \
                awk '{print $9}' | \
                head -n ${ipeer} | tail -n 1)
         # Check if actually used message size is consistent
         # with expected message size
         if [[ "${count}" -ne "${nb}" ]] ; then
            echo "Message send size from rank ${irank} to ${jrank} is ${count}!"
            echo "Was expecting message size of ${nb}!"
            exit 1;
         fi
         # Check if actually used peer process is consistent
         # with expected peer process
         if [[ "${peer}" -ne "${jrank}" ]] ; then
            echo "Message send from rank ${irank} to ${peer}!"
            echo "Was expecting sending to rank ${jrank}!"
            exit 1;
         fi
         
         # validate receiving
         # check if every message appears the correct amount of times
         msgcount=$(../../../tools/vftrace_vfd_dump ${vftr_binary}_${irank}.vfd | \
                    awk '$2=="recv" && $3!="end"{getline;print;}' | \
                    grep "peer=${jrank}" | \
                    wc -l)
         if [[ "${msgcount}" -ne "${msgnum_val}" ]] ; then
            echo "Message received by rank ${irank} from ${jrank} appeared ${msgcount} times!"
            echo "Was expecting message to appear ${msgnum_val} times!"
            exit 1;
         fi
         # Get actually used message size
         count=$(../../../tools/vftrace_vfd_dump ${vftr_binary}_${irank}.vfd | \
                 awk '$2=="recv" && $3!="end"{getline;print;}' | \
                 grep "peer=${jrank}" | \
                 sed 's/=/ /g' | \
                 awk '{print $2}' | \
                 head -n ${ipeer} | tail -n 1)
         # get peer process
         peer=$(../../../tools/vftrace_vfd_dump ${vftr_binary}_${irank}.vfd | \
                awk '$2=="recv" && $3!="end"{getline;print;}' | \
                grep "peer=${jrank}" | \
                sed 's/=/ /g' | \
                awk '{print $9}' | \
                head -n ${ipeer} | tail -n 1)
         # Check if actually used message size is consistent
         # with expected message size
         if [[ "${count}" -ne "${nb}" ]] ; then
            echo "Message recv size from rank ${jrank} to ${irank} is ${count}!"
            echo "Was expecting message size of ${nb}!"
            exit 1;
         fi
         # Check if actually used peer process is consistent
         # with expected peer process
         if [[ "${peer}" -ne "${jrank}" ]] ; then
            echo "Message received from rank ${peer} by ${irank}!"
            echo "Was expecting receiving from rank ${jrank}!"
            exit 1;
         fi
      done
   done
done
