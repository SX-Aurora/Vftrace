#!/bin/bash

source ${srcdir}/../../environment/filenames.sh

vftr_binary=ialltoallw_intercom
configfile=${vftr_binary}.json
nprocs=4
ntrials=1

determine_bin_prefix $vftr_binary

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
      vfdfile=$(get_vfdfile_name ${vftr_binary} ${irank})
      ../../../tools/vftrace_vfd_dump ${vfdfile}

      my_group=$(bc <<< "(2*${irank})/${nprocs}")
      # validate receive
      minpeerrank=$(bc <<< "(1-${my_group})*((${nprocs}+1)/2)")
      maxpeerrank=$(bc <<< "${minpeerrank} + (${nprocs}+${my_group})/2 -1")

      ipeer=0
      for jrank in $(seq ${minpeerrank} 1 ${maxpeerrank});
      do
         ((ipeer+=1))
         tmpnb=$(bc <<< "${nb}+${irank}")
         # the sending group
         # Get actually used message size
         count=$(../../../tools/vftrace_vfd_dump ${vfdfile} | \
                 awk '$2=="send" && $3!="end"{getline;print;}' | \
                 sed 's/=/ /g' | \
                 sort -nk 9 | \
                 awk '{print $2}' | \
                 head -n ${ipeer} | tail -n 1)
         # get peer process
         peer=$(../../../tools/vftrace_vfd_dump ${vfdfile} | \
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

         tmpnb=$(bc <<< "${nb}+${jrank}")
         # Validate receiving
         # Get actually used message size
         count=$(../../../tools/vftrace_vfd_dump ${vfdfile} | \
                 awk '$2=="recv" && $3!="end"{getline;print;}' | \
                 sed 's/=/ /g' | \
                 sort -nk 9 | \
                 awk '{print $2}' | \
                 head -n ${ipeer} | tail -n 1)
         # get peer process
         peer=$(../../../tools/vftrace_vfd_dump ${vfdfile} | \
                awk '$2=="recv" && $3!="end"{getline;print;}' | \
                sed 's/=/ /g' | \
                sort -nk 9 | \
                awk '{print $9}' | \
                head -n ${ipeer} | tail -n 1)
         # Check if actually used message size is consistent
         # with expected message size
         if [[ "${count}" -ne "${tmpnb}" ]] ; then
            echo "Message recv size from rank ${jrank} by ${irank} is ${count}!"
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
