#!/bin/bash

vftr_binary=scatterv_intercom
configfile=${vftr_binary}.json
nprocs=4
ntrials=1

echo "{\"sampling\": {\"active\": true}, \"mpi\": {\"active\": true}}" > ${configfile}
export VFTR_CONFIG=${configfile}

for itrial in $(seq 1 1 ${ntrials});
do
   # Generate a random message size
   nb=$(bc <<< "32*${RANDOM}")
   ${MPI_EXEC} ${MPI_OPTS} ${NP} ${nprocs} ./${vftr_binary} ${nb} || exit 1

   root_proc=0
   # check each rank for the correct message communication
   # patterns in the vfd file
   for irank in $(seq 0 1 $(bc <<< "${nprocs}-1"));
   do
      ../../../tools/vftrace_vfd_dump ${vftr_binary}_${irank}.vfd

      my_group=$(bc <<< "(2*${irank})/${nprocs}")
      if [ "${my_group}" -eq "0" ] ; then
         # The sending group
         if [ "${irank}" -eq "0" ] ; then
            # The sending rank in the sending group
            minrecvrank=$(bc <<< "(${nprocs}+1)/2")
            maxrecvrank=$(bc <<< "(${nprocs}-1)")
            ipeer=0
            for jrank in $(seq ${minrecvrank} 1 ${maxrecvrank});
            do
               tmpnb=$(bc <<< "${nb}+${jrank}")
               ((ipeer+=1))
               # Validate sending
               # Get actually used message size
               count=$(../../../tools/vftrace_vfd_dump ${vftr_binary}_${irank}.vfd | \
                       awk '$2=="send" && $3!="end"{getline;print;}' | \
                       sed 's/=/ /g' | \
                       sort -nk 9 | \
                       awk '{print $2}' | \
                       head -n ${ipeer} | tail -n 1)
               # get peer process
               peer=$(../../../tools/vftrace_vfd_dump ${vftr_binary}_${irank}.vfd | \
                      awk '$2=="send" && $3!="end"{getline;print;}' | \
                      sed 's/=/ /g' | \
                      sort -nk 9 | \
                      awk '{print $9}' | \
                      head -n ${ipeer} | tail -n 1)
               # Check if actually used message size is consistent
               # with expected message size
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
            done
         else
            # Non participating processes in the receiving group
            # must not show any communication
            ncom=$(../../../tools/vftrace_vfd_dump ${vftr_binary}_${irank}.vfd | \
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
         tmpnb=$(bc <<< "${nb} + ${irank}")
         # Get actually used message size
         count=$(../../../tools/vftrace_vfd_dump ${vftr_binary}_${irank}.vfd | \
                 awk '$2=="recv" && $3!="end"{getline;print;}' | \
                 sed 's/=/ /g' | \
                 sort -nk 9 | \
                 awk '{print $2}' | \
                 head -n 1)
         # get peer process
         peer=$(../../../tools/vftrace_vfd_dump ${vftr_binary}_${irank}.vfd | \
                awk '$2=="recv" && $3!="end"{getline;print;}' | \
                sed 's/=/ /g' | \
                sort -nk 9 | \
                awk '{print $9}' | \
                head -n 1)
         # Check if actually used message size is consistent
         # with expected message size
         if [[ "${count}" -ne "${tmpnb}" ]] ; then
            echo "Message receive size of rank ${irank} from 0 is ${count}!"
            echo "Was expecting message size of ${tmpnb}!"
            exit 1;
         fi
         # Check if actually used peer process is consistent
         # with expected peer process
         if [[ "${peer}" -ne "0" ]] ; then
            echo "Message received on rank ${irank} from ${peer}!"
            echo "Was expecting receiving from rank 0!"
            exit 1;
         fi
      fi
   done
done
