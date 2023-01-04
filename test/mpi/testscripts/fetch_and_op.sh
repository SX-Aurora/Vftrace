#!/bin/bash

source ${srcdir}/../../environment/filenames.sh

vftr_binary=fetch_and_op
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

   vfdfile=$(get_vfdfile_name ${vftr_binary} 0)
   ../../../tools/vftrace_vfd_dump ${vfdfile}
   for irank in $(seq 1 1 $(bc <<< "${nprocs}-1"));
   do
     # Validate receiving
     # Get actually used message size
     count=$(../../../tools/vftrace_vfd_dump ${vfdfile} | \
             awk '$2=="recv" && $3!="end"{getline;print;}' | \
             sed 's/=/ /g' | \
             sort -nk 9 | \
             awk '{print $2}' | \
             head -n ${irank} | tail -n 1)
     # get peer process
     peer=$(../../../tools/vftrace_vfd_dump ${vfdfile} | \
            awk '$2=="recv" && $3!="end"{getline;print;}' | \
            sed 's/=/ /g' | \
            sort -nk 9 | \
            awk '{print $9}' | \
            head -n ${irank} | tail -n 1)
     # Check if actually used message size is consistent
     # with expected message size
     if [[ "${count}" -ne "1" ]] ; then
        echo "Message recv size from rank ${irank} to 0 is ${count}!"
        echo "Was expecting message size of 1!"
        exit 1;
     fi
     # Check if actually used peer process is consistent
     # with expected peer process
     if [[ "${peer}" -ne "${irank}" ]] ; then
        echo "Message received from rank ${peer} by 0!"
        echo "Was expecting receiving from rank ${irank}!"
        exit 1;
     fi

     # Validate sending
     # Get actually used message size
     count=$(../../../tools/vftrace_vfd_dump ${vfdfile} | \
             awk '$2=="send" && $3!="end"{getline;print;}' | \
             sed 's/=/ /g' | \
             sort -nk 9 | \
             awk '{print $2}' | \
             head -n ${irank} | tail -n 1)
     # get peer process
     peer=$(../../../tools/vftrace_vfd_dump ${vfdfile} | \
            awk '$2=="send" && $3!="end"{getline;print;}' | \
            sed 's/=/ /g' | \
            sort -nk 9 | \
            awk '{print $9}' | \
            head -n ${irank} | tail -n 1)
     # Check if actually used message size is consistent
     # with expected message size
     if [[ "${count}" -ne "1" ]] ; then
        echo "Message send size from rank 0 to ${irank} is ${count}!"
        echo "Was expecting message size of 1!"
        exit 1;
     fi
     # Check if actually used peer process is consistent
     # with expected peer process
     if [[ "${peer}" -ne "${irank}" ]] ; then
        echo "Message send from rank 0 to ${peer}!"
        echo "Was expecting sending to rank ${irank}!"
        exit 1;
     fi
   done
done
