#!/bin/bash

source ${srcdir}/../../environment/filenames.sh

vftr_binary=reduce_scatter_block
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
   sbufsize=$(bc <<< "${nprocs}*${nb}")
   for irank in $(seq 0 1 $(bc <<< "${nprocs}-1"));
   do
      vfdfile=$(get_vfdfile_name ${vftr_binary} ${irank})
      ../../../tools/vftrace_vfd_dump ${vfdfile}

      # The 0-th rank performs a reduction and
      # scatters the result to all other ranks
      if [[ ${irank} -eq "0" ]] ; then
         for jrank in $(seq 0 1 $(bc <<< "${nprocs}-1"));
         do
            ipeer=$(bc <<< ${jrank}+1)
            # Validate reduction receive
            # Get actually used message size
            count=$(../../../tools/vftrace_vfd_dump ${vfdfile} | \
                    awk '$2=="recv" && $3!="end"{getline;print;}' | \
                    sed 's/=/ /g' | \
                    sort -nk 9 | sort -rsnk 2 | \
                    awk '{print $2}' | \
                    head -n ${ipeer} | tail -n 1)
            # get peer process
            peer=$(../../../tools/vftrace_vfd_dump ${vfdfile} | \
                   awk '$2=="recv" && $3!="end"{getline;print;}' | \
                   sed 's/=/ /g' | \
                   sort -nk 9 | sort -rsnk 2 | \
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
               echo "Was expecting recieving from rank ${jrank}!"
               exit 1;
            fi

            # validate scatter send
            count=$(../../../tools/vftrace_vfd_dump ${vfdfile} | \
                    awk '$2=="send" && $3!="end"{getline;print;}' | \
                    sed 's/=/ /g' | \
                    sort -nk 9 | sort -snk 2 | \
                    awk '{print $2}' | \
                    head -n ${ipeer} | tail -n 1)
            peer=$(../../../tools/vftrace_vfd_dump ${vfdfile} | \
                   awk '$2=="send" && $3!="end"{getline;print;}' | \
                   sed 's/=/ /g' | \
                   sort -nk 9 | sort -snk 2 | \
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
         done
      fi

      jrank=0
      ipeer=1
      # Validate send to root rank for reduction
      count=$(../../../tools/vftrace_vfd_dump ${vfdfile} | \
              awk '$2=="send" && $3!="end"{getline;print;}' | \
              sed 's/=/ /g' | \
              sort -nk 9 | sort -srnk 2 | \
              awk '{print $2}' | \
              head -n ${ipeer} | tail -n 1)
      peer=$(../../../tools/vftrace_vfd_dump ${vfdfile} | \
             awk '$2=="send" && $3!="end"{getline;print;}' | \
             sed 's/=/ /g' | \
             sort -nk 9 | sort -srnk 2 | \
             awk '{print $9}' | \
             head -n ${ipeer} | tail -n 1)
      # Check if actually used message size is consistent
      # with expected message size
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

      # validate receive from root rank as scatter
      count=$(../../../tools/vftrace_vfd_dump ${vfdfile} | \
              awk '$2=="recv" && $3!="end"{getline;print;}' | \
              sed 's/=/ /g' | \
              sort -nk 9 | sort -snk 2 | \
              awk '{print $2}' | \
              head -n ${ipeer} | tail -n 1)
      peer=$(../../../tools/vftrace_vfd_dump ${vfdfile} | \
             awk '$2=="recv" && $3!="end"{getline;print;}' | \
             sed 's/=/ /g' | \
             sort -nk 9 | sort -snk 2 | \
             awk '{print $9}' | \
             head -n ${ipeer} | tail -n 1)
      # Check if actually used message size is consistent
      # with expected message size
      if [[ "${count}" -ne "${nb}" ]] ; then
         echo "Message recv size from rank ${irank} by ${jrank} is ${count}!"
         echo "Was expecting message size of ${nb}!"
         exit 1;
      fi
      # Check if actually used peer process is consistent
      # with expected peer process
      if [[ "${peer}" -ne "${jrank}" ]] ; then
         echo "Message recv from rank ${peer} by ${irank}!"
         echo "Was expecting receiving from rank ${jrank}!"
         exit 1;
      fi
   done
done
