#!/bin/bash

vftr_binary=put
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

   ../../../tools/vftrace_vfd_dump ${vftr_binary}_0.vfd
   for irank in $(seq 1 1 $(bc <<< "${nprocs}-1"));
   do
     # Validate receiving
     # Get actually used message size
     count=$(../../../tools/vftrace_vfd_dump ${vftr_binary}_0.vfd | \
             awk '$2=="send" && $3!="end"{getline;print;}' | \
             sed 's/=/ /g' | \
             sort -nk 9 | \
             awk '{print $2}' | \
             head -n ${irank} | tail -n 1)
     # get peer process
     peer=$(../../../tools/vftrace_vfd_dump ${vftr_binary}_0.vfd | \
            awk '$2=="send" && $3!="end"{getline;print;}' | \
            sed 's/=/ /g' | \
            sort -nk 9 | \
            awk '{print $9}' | \
            head -n ${irank} | tail -n 1)
     # Check if actually used message size is consistent
     # with expected message size
     if [[ "${count}" -ne "${nb}" ]] ; then
        echo "Message send size from rank ${irank} to 0 is ${count}!"
        echo "Was expecting message size of ${nb}!"
        exit 1;
     fi
     # Check if actually used peer process is consistent
     # with expected peer process
     if [[ "${peer}" -ne "${irank}" ]] ; then
        echo "Message received from rank ${peer} by 0!"
        echo "Was expecting receiving from rank ${irank}!"
        exit 1;
     fi
   done
done
