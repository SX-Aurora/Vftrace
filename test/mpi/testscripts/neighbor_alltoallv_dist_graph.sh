#!/bin/bash

vftr_binary=neighbor_alltoallv_dist_graph
nprocs=4
ntrials=1

export VFTR_SAMPLING="Yes"
export VFTR_MPI_LOG="Yes"

for itrial in $(seq 1 1 ${ntrials});
do
   # Generate a random message size
   nb=$(bc <<< "32*${RANDOM}")
   ${MPI_EXEC} ${MPI_OPTS} ${NP} ${nprocs} ./${vftr_binary} ${nb} || exit 1

   # A neighbor list needs to be created for checking the communication
   sneighborlist[0]="1 2"
   sneighborlist[1]="0 2"
   sneighborlist[2]="0 3"
   sneighborlist[3]="3"
   smsgnum[0]="2 1"
   smsgnum[1]="2 1"
   smsgnum[2]="1 1"
   smsgnum[3]="1"
   rneighborlist[0]="1 2"
   rneighborlist[1]="0"
   rneighborlist[2]="0 1"
   rneighborlist[3]="2 3"
   rmsgnum[0]="2 1"
   rmsgnum[1]="2"
   rmsgnum[2]="1 1"
   rmsgnum[3]="1 1"
   # check each rank for the correct message communication
   # patterns in the vfd file
   for irank in $(seq 0 1 $(bc <<< "${nprocs}-1"));
   do
      ../../../tools/vftrace_vfd_dump ${vftr_binary}_${irank}.vfd

      nspeers=$(echo "${sneighborlist[${irank}]}" | wc -w)
      totsendmsg=$(echo "${smsgnum[${irank}]}" | sed 's/ /+/g' | bc)
      nrpeers=$(echo "${rneighborlist[${irank}]}" | wc -w)
      totrecvmsg=$(echo "${rmsgnum[${irank}]}" | sed 's/ /+/g' | bc)

      # check if the total number of messages is the sum of individual ones
      totmsgcount=$(../../../tools/vftrace_vfd_dump ${vftr_binary}_${irank}.vfd | \
                    awk '$2=="send" && $3!="end"{getline;print;}' | \
                    wc -l)
      if [[ "${totmsgcount}" -ne "${totsendmsg}" ]] ; then
         echo "Expected a total of ${totsendmsg} messages to be send from rank ${irank}!"
         echo "Found a total of ${totmsgcount} messages!"
         exit 1;
      fi
      totmsgcount=$(../../../tools/vftrace_vfd_dump ${vftr_binary}_${irank}.vfd | \
                    awk '$2=="recv" && $3!="end"{getline;print;}' | \
                    wc -l)
      if [[ "${totmsgcount}" -ne "${totrecvmsg}" ]] ; then
         echo "Expected a total of ${totrecvmsg} messages to be received by rank ${irank}!"
         echo "Found a total of ${totmsgcount} messages!"
         exit 1;
      fi

      for peer_idx in $(seq 1 1 ${npeers});
      do
         jrank=$(echo "${sneighborlist[${irank}]}" | awk -v i=${peer_idx} '{print $i}')
         msgnum_val=$(echo "${smsgnum[${irank}]}" | awk -v i=${peer_idx} '{print $i}')
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
         tmpnb=$(echo "${nb} + ${irank}" | bc)
         if [[ "${count}" -ne "${tmpnb}" ]] ; then
            echo "Message send size from rank ${irank} to ${jrank} is ${count}!"
            echo "Was expecting message size of ${tmpnb}!"
            exit 1;
         fi
         # Check if actually used peer process is consistent
         # with expected peer process
         if [[ "${peer}" -ne "${jrank}" ]] ; then
            echo "Message send from rank ${irank} to ${peer}!"
            echo "Was expecting sending to rank ${jrank}!"
            exit 1;
         fi
         
         jrank=$(echo "${rneighborlist[${irank}]}" | awk -v i=${peer_idx} '{print $i}')
         msgnum_val=$(echo "${rmsgnum[${irank}]}" | awk -v i=${peer_idx} '{print $i}')
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
         tmpnb=$(echo "${nb} + ${jrank}" | bc)
         if [[ "${count}" -ne "${tmpnb}" ]] ; then
            echo "Message recv size from rank ${jrank} to ${irank} is ${count}!"
            echo "Was expecting message size of ${tmpnb}!"
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
