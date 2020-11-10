#!/bin/bash

vftr_binary=iscan_inplace
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

   # check each rank for the correct message communication
   # patterns in the vfd file
   maxrank=$(bc <<< "${nprocs}-1")
   for irank in $(seq 0 1 ${maxrank});
   do
      ../../../tools/tracedump ${vftr_binary}_${irank}.vfd
      if [ "${irank}" -eq "0" ] ; then
         # Validate sending
         # Rank 0 sends to rank 1
         jrank=$(bc <<< "${irank} + 1")
         # Get actually used message size
         count=$(../../../tools/tracedump ${vftr_binary}_${irank}.vfd | \
                 awk '$2=="send" && $3!="end"{getline;print;}' | \
                 sed 's/=/ /g' | \
                 sort -nk 9 | \
                 awk '{print $2}' | \
                 head -n 2 | tail -n 1)
         # get peer process
         peer=$(../../../tools/tracedump ${vftr_binary}_${irank}.vfd | \
                awk '$2=="send" && $3!="end"{getline;print;}' | \
                sed 's/=/ /g' | \
                sort -nk 9 | \
                awk '{print $9}' | \
                head -n 2 | tail -n 1)
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
      elif [ "${irank}" -eq "${maxrank}" ] ; then
         # Validate receiving
         # Last rank only receives from lastrank-1
         jrank=$(bc <<< "${irank} - 1")
         # Get actually used message size
         count=$(../../../tools/tracedump ${vftr_binary}_${irank}.vfd | \
                 awk '$2=="recv" && $3!="end"{getline;print;}' | \
                 sed 's/=/ /g' | \
                 sort -nk 9 | \
                 awk '{print $2}' | \
                 head -n 1)
         # get peer process
         peer=$(../../../tools/tracedump ${vftr_binary}_${irank}.vfd | \
                awk '$2=="recv" && $3!="end"{getline;print;}' | \
                sed 's/=/ /g' | \
                sort -nk 9 | \
                awk '{print $9}' | \
                head -n 1)
         # Check if actually used message size is consistent
         # with expected message size
         if [[ "${count}" -ne "${nb}" ]] ; then
            echo "Message receive size on rank ${irank} from ${jrank} is ${count}!"
            echo "Was expecting message size of ${nb}!"
            exit 1;
         fi
         # Check if actually used peer process is consistent
         # with expected peer process
         if [[ "${peer}" -ne "${jrank}" ]] ; then
            echo "Message received on rank ${irank} from ${peer}!"
            echo "Was expecting receiving from rank ${jrank}!"
            exit 1;
         fi
      else
         # all other ranks
         jrank=$(bc <<< "${irank} + 1")
         # Validate sending
         # Get actually used message size
         count=$(../../../tools/tracedump ${vftr_binary}_${irank}.vfd | \
                 awk '$2=="send" && $3!="end"{getline;print;}' | \
                 sed 's/=/ /g' | \
                 sort -nk 9 | \
                 awk '{print $2}' | \
                 head -n 1)
         # get peer process
         peer=$(../../../tools/tracedump ${vftr_binary}_${irank}.vfd | \
                awk '$2=="send" && $3!="end"{getline;print;}' | \
                sed 's/=/ /g' | \
                sort -nk 9 | \
                awk '{print $9}' | \
                head -n 1)
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
         jrank=$(bc <<< "${irank} - 1")
         # Validate receiving
         # Get actually used message size
         count=$(../../../tools/tracedump ${vftr_binary}_${irank}.vfd | \
                 awk '$2=="recv" && $3!="end"{getline;print;}' | \
                 sed 's/=/ /g' | \
                 sort -nk 9 | \
                 awk '{print $2}' | \
                 head -n 1)
         # get peer process
         peer=$(../../../tools/tracedump ${vftr_binary}_${irank}.vfd | \
                awk '$2=="recv" && $3!="end"{getline;print;}' | \
                sed 's/=/ /g' | \
                sort -nk 9 | \
                awk '{print $9}' | \
                head -n 1)
         # Check if actually used message size is consistent
         # with expected message size
         if [[ "${count}" -ne "${nb}" ]] ; then
            echo "Message receive size on rank ${irank} from ${jrank} is ${count}!"
            echo "Was expecting message size of ${nb}!"
            exit 1;
         fi
         # Check if actually used peer process is consistent
         # with expected peer process
         if [[ "${peer}" -ne "${jrank}" ]] ; then
            echo "Message received on rank ${irank} from ${peer}!"
            echo "Was expecting receiving from rank ${jrank}!"
            exit 1;
         fi
      fi
   done
done
