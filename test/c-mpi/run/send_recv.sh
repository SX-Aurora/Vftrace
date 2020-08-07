#!/bin/bash

vftr_binary=send_recv
nprocs=2
ntrials=1

export VFTR_SAMPLING="Yes"
export VFTR_MPI_LOG="Yes"
export VFTR_PRECISE="MPI_*"

for itrial in $(seq 1 1 ${ntrials});
do
   nb=$(bc <<< "32*${RANDOM}")
   mpirun -np ${nprocs} ./${vftr_binary} ${nb}

   for irank in $(seq 0 1 $(bc <<< "${nprocs}-1"));
   do
      npeers=0
      ../../tools/tracedump ${vftr_binary}_${irank}.vfd
      for jrank in $(seq 0 1 $(bc <<< "${nprocs}-1"));
      do
         if [[ ${irank} -ne ${jrank} ]] ; then
            ((npeers+=1))
            # Validate sending
            count=$(../../tools/tracedump ${vftr_binary}_${irank}.vfd | \
                    awk '$2=="send" && $3!="end"{getline;print;}' | \
                    sed 's/=/ /g' | \
                    awk '{print $2}' | \
                    head -n ${npeers} | tail -n 1)
            peer=$(../../tools/tracedump ${vftr_binary}_${irank}.vfd | \
                   awk '$2=="send" && $3!="end"{getline;print;}' | \
                   sed 's/=/ /g' | \
                   awk '{print $9}' | \
                   head -n ${npeers} | tail -n 1)
            if [[ "${count}" -ne "${nb}" ]] ; then
               echo "Message send size from rank ${irank} to ${jrank} is ${count}!"
               echo "Was expecting message size of ${nb}!"
               exit 1;
            fi
            if [[ "${peer}" -ne "${jrank}" ]] ; then
               echo "Message send from rank ${irank} to ${peer}!"
               echo "Was expecting sending to rank ${jrank}!"
               exit 1;
            fi

            # Validate receiving
            count=$(../../tools/tracedump ${vftr_binary}_${irank}.vfd | \
                    awk '$2=="recv" && $3!="end"{getline;print;}' | \
                    sed 's/=/ /g' | \
                    awk '{print $2}' | \
                    head -n ${npeers} | tail -n 1)
            peer=$(../../tools/tracedump ${vftr_binary}_${irank}.vfd | \
                   awk '$2=="recv" && $3!="end"{getline;print;}' | \
                   sed 's/=/ /g' | \
                   awk '{print $9}' | \
                   head -n ${npeers} | tail -n 1)

            if [[ "${count}" -ne "${nb}" ]] ; then
               echo "Message recv size from rank ${jrank} to ${irank} is ${count}!"
               echo "Was expecting message size of ${nb}!"
               exit 1;
            fi
            if [[ "${peer}" -ne "${jrank}" ]] ; then
               echo "Message received from rank ${peer} by ${irank}!"
               echo "Was expecting receiving from rank ${jrank}!"
               exit 1;
            fi
         fi
      done
   done
done
