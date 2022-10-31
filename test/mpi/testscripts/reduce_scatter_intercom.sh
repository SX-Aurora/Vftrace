#!/bin/bash

vftr_binary=reduce_scatter_intercom
configfile=${vftr_binary}.json
nprocs=4
ntrials=1

echo "{\"sampling\": {\"active\": true}}" > ${configfile}
export VFTR_CONFIG=${configfile}

for itrial in $(seq 1 1 ${ntrials});
do
   # Generate a random message size
   nb=$(bc <<< "32*${RANDOM}")
   ${MPI_EXEC} ${MPI_OPTS} ${NP} ${nprocs} ./${vftr_binary} ${nb} || exit 1

   # check each rank for the correct message communication
   # patterns in the vfd file
   for irank in $(seq 0 1 $(bc <<< "${nprocs}-1"));
   do

      ../../../tools/vftrace_vfd_dump ${vftr_binary}_${irank}.vfd

      my_group=$(bc <<< "(2*${irank})/${nprocs}")
      minlocalpeerrank=$(bc <<< "${my_group}*((${nprocs}+1)/2)")
      minremotepeerrank=$(bc <<< "(1-${my_group})*((${nprocs}+1)/2)")
      maxremotepeerrank=$(bc <<< "${minremotepeerrank} + (${nprocs}+${my_group})/2 -1")
      remote_group_size=$(bc <<< "1 + ${maxremotepeerrank} - ${minremotepeerrank}")
      local_group_size=$(bc <<< "${nprocs} - ${remote_group_size}")
      maxlocalpeerrank=$(bc <<< "${minlocalpeerrank} + ${local_group_size} - 1")

      sbufsize=$(bc <<< "${local_group_size}*${remote_group_size}*${nb}")
      recvnb=$(bc <<< "${nb} * ${remote_group_size}")

      # First every process sends the complete sendbuffer to the 0 process of the remote group
      jrank=${minremotepeerrank}
      # Get the actually used message size
      count=$(../../../tools/vftrace_vfd_dump ${vftr_binary}_${irank}.vfd | \
              awk '$2=="send" && $3!="end"{getline;print;}' | \
              sed 's/=/ /g' | \
              sort -srnk 2 | \
              awk '{print $2}' | \
              head -n 1 | tail -n 1)
      #get peer process
      peer=$(../../../tools/vftrace_vfd_dump ${vftr_binary}_${irank}.vfd | \
             awk '$2=="send" && $3!="end"{getline;print;}' | \
             sed 's/=/ /g' | \
             sort -srnk 2 | \
             awk '{print $9}' | \
             head -n 1 | tail -n 1)
      if [[ "${count}" -ne "${sbufsize}" ]] ; then
         echo "Message send size from rank ${irank} to ${jrank} is ${count}!"
         echo "Was expecting message size of ${sbufsize}!"
         exit 1;
      fi
      # Check if actually used peer process is consistent
      # with expected peer process
      if [[ "${peer}" -ne "${jrank}" ]] ; then
         echo "Message send from rank ${irank} to ${peer}!"
         echo "Was expecting sending to rank ${jrank}!"
         exit 1;
      fi

      # If the current process is process 0 of a group it receives
      # the complete sendbuffer from every process of the remote group for reduction
      if [[ ${irank} -eq ${minlocalpeerrank} ]] ; then
         ipeer=0
         for jrank in $(seq ${minremotepeerrank} 1 ${maxremotepeerrank});
         do
            ((ipeer+=1))
            # Get the actually used message size
            count=$(../../../tools/vftrace_vfd_dump ${vftr_binary}_${irank}.vfd | \
                    awk '$2=="recv" && $3!="end"{getline;print;}' | \
                    sed 's/=/ /g' | \
                    sort -nk 9 | sort -srnk 2 | \
                    awk '{print $2}' | \
                    head -n ${ipeer} | tail -n 1)
            #get peer process
            peer=$(../../../tools/vftrace_vfd_dump ${vftr_binary}_${irank}.vfd | \
                   awk '$2=="recv" && $3!="end"{getline;print;}' | \
                   sed 's/=/ /g' | \
                   sort -nk 9 | sort -srnk 2 | \
                   awk '{print $9}' | \
                   head -n ${ipeer} | tail -n 1)
            # Check if actually used message size is consistent
            # with expected message size
            if [[ "${count}" -ne "${sbufsize}" ]] ; then
               echo "Message recv size from rank ${jrank} by ${irank} is ${count}!"
               echo "Was expecting message size of ${sbufsize}!"
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
      fi

      # If the current process is process 0 of a group it scatters
      # the recently reduce data to the members of its group
      if [[ ${irank} -eq ${minlocalpeerrank} ]] ; then
         ipeer=0
         for jrank in $(seq ${minlocalpeerrank} 1 ${maxlocalpeerrank});
         do
            # compute jrank specific size
            tmpsize=$(bc <<< "${sbufsize} / ${local_group_size} - ${local_group_size} / 2 + ${jrank}-${minlocalpeerrank}")
            tmpsize=$(bc <<< "${tmpsize} + (${jrank}-${minlocalpeerrank} >= ${local_group_size}/2) * ((${local_group_size}+1)%2)")
            ((ipeer+=1))
            # Get the actually used message size
            count=$(../../../tools/vftrace_vfd_dump ${vftr_binary}_${irank}.vfd | \
                    awk '$2=="send" && $3!="end"{getline;print;}' | \
                    sed 's/=/ /g' | \
                    sort -nk 2 | \
                    awk '{print $2}' | \
                    head -n ${ipeer} | tail -n 1)
            #get peer process
            peer=$(../../../tools/vftrace_vfd_dump ${vftr_binary}_${irank}.vfd | \
                   awk '$2=="send" && $3!="end"{getline;print;}' | \
                   sed 's/=/ /g' | \
                   sort -nk 2 | \
                   awk '{print $9}' | \
                   head -n ${ipeer} | tail -n 1)
            # Check if actually used message size is consistent
            # with expected message size
            if [[ "${count}" -ne "${tmpsize}" ]] ; then
               echo "Message send size from rank ${irank} to ${jrank} is ${count}!"
               echo "Was expecting message size of ${tmpsize}!"
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
      fi

      # receive the reduce data scattered by process 0 of the local group
      jrank=${minlocalpeerrank}
      # compute jrank specific size
      tmpsize=$(bc <<< "${sbufsize} / ${local_group_size} - ${local_group_size} / 2 + ${irank}-${minlocalpeerrank}")
      tmpsize=$(bc <<< "${tmpsize} + (${irank}-${minlocalpeerrank} >= ${local_group_size}/2) * ((${local_group_size}+1)%2)")
      # Get the actually used message size
      count=$(../../../tools/vftrace_vfd_dump ${vftr_binary}_${irank}.vfd | \
              awk '$2=="recv" && $3!="end"{getline;print;}' | \
              sed 's/=/ /g' | \
              sort -nk 2 | \
              awk '{print $2}' | \
              head -n 1 | tail -n 1)
      #get peer process
      peer=$(../../../tools/vftrace_vfd_dump ${vftr_binary}_${irank}.vfd | \
             awk '$2=="recv" && $3!="end"{getline;print;}' | \
             sed 's/=/ /g' | \
             sort -nk 2 | \
             awk '{print $9}' | \
             head -n 1 | tail -n 1)
      # Check if actually used message size is consistent
      # with expected message size
      if [[ "${count}" -ne "${tmpsize}" ]] ; then
         echo "Message recv size from rank ${jrank} by ${irank} is ${count}!"
         echo "Was expecting message size of ${tmpsize}!"
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
