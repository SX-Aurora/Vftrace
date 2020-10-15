#!/bin/bash

vftr_binary=iscatter_intercom
nprocs=4
ntrials=1

export VFTR_SAMPLING="Yes"
export VFTR_MPI_LOG="Yes"
export VFTR_PRECISE="MPI_*|mpi_*"

for itrial in $(seq 1 1 ${ntrials});
do
   # Generate a random message size
   nb=$(bc <<< "32*${RANDOM}")
   mpirun -np ${nprocs} ./${vftr_binary} ${nb} || exit 1

   root_proc=0
   # check each rank for the correct message communication
   # patterns in the vfd file
   for irank in $(seq 0 1 $(bc <<< "${nprocs}-1"));
   do
      ../../../tools/tracedump ${vftr_binary}_${irank}.vfd

      my_group=$(bc <<< "(2*${irank})/${nprocs}")
      if [ "${my_group}" -eq "0" ] ; then
         # The sending group
         if [ "${irank}" -eq "0" ] ; then
            # The sending rank in the sending group
            minsendrank=$(bc <<< "(${nprocs}+1)/2")
            maxsendrank=$(bc <<< "(${nprocs}-1)")
            ipeer=0
            for jrank in $(seq ${minsendrank} 1 ${maxsendrank});
            do
               ((ipeer+=1))
               # Validate receiving
               # Get actually used message size
               count=$(../../../tools/tracedump ${vftr_binary}_${irank}.vfd | \
                       awk '$2=="send" && $3!="end"{getline;print;}' | \
                       sed 's/=/ /g' | \
                       sort -nk 9 | \
                       awk '{print $2}' | \
                       head -n ${ipeer} | tail -n 1)
               # get peer process
               peer=$(../../../tools/tracedump ${vftr_binary}_${irank}.vfd | \
                      awk '$2=="send" && $3!="end"{getline;print;}' | \
                      sed 's/=/ /g' | \
                      sort -nk 9 | \
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
                  echo "Message send to rank ${peer} by ${irank}!"
                  echo "Was expecting sending to rank ${jrank}!"
                  exit 1;
               fi
            done
         else
            # Non participating processes in the sending group
            # must not show any communication
            ncom=$(../../../tools/tracedump ${vftr_binary}_${irank}.vfd | \
                   awk '($2=="send" || $2=="recv") && $3!="end"{getline;print;}' | \
                   wc -l)
            if [[ "${ncomm}" -ne "0" ]] ; then
               echo "Process ${irank} participated in communication!"
               echo "It was not supposed to!"
               exit 1;
            fi
         fi
      else
         # the receiving group
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
            echo "Message receive size from rank 0 by ${irank} is ${count}!"
            echo "Was expecting message size of ${nb}!"
            exit 1;
         fi
         # Check if actually used peer process is consistent
         # with expected peer process
         if [[ "${peer}" -ne "0" ]] ; then
            echo "Message received by rank ${irank} from ${peer}!"
            echo "Was expecting to receive from rank 0!"
            exit 1;
         fi
      fi
   done
done
