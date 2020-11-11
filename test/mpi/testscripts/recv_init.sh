#!/bin/bash

vftr_binary=recv_init
nprocs=4
ntrials=1

export VFTR_SAMPLING="Yes"
export VFTR_MPI_LOG="Yes"
export VFTR_PRECISE="MPI_*|mpi_*"

for itrial in $(seq 1 1 ${ntrials});
do
   # Generate a random message size
   nb=$(bc <<< "32*${RANDOM}")
   nr=$(bc <<< "(${RANDOM}%5)+5")
   ${MPI_EXEC} ${MPI_OPTS} ${NP} ${nprocs} ./${vftr_binary} ${nb} ${nr} || exit 1

   # check each rank for the correct message communication
   # patterns in the vfd file
   for irank in $(seq 0 1 $(bc <<< "${nprocs}-1"));
   do

      ../../../tools/tracedump ${vftr_binary}_${irank}.vfd

      if [[ "${irank}" -eq "0" ]] ; then 

         for jrank in $(seq 1 1 $(bc <<< "${nprocs}-1"));
         do
            for irun in $(seq 1 1 ${nr});
            do
               ientry=$(bc <<< "(${jrank}-1)*${nr} + ${irun}")
               # Validate receiving
               # Get actually used message size
               count=$(../../../tools/tracedump ${vftr_binary}_${irank}.vfd | \
                       awk '$2=="recv" && $3!="end"{getline;print;}' | \
                       sed 's/=/ /g' | \
                       sort -nk 9 | \
                       awk '{print $2}' | \
                       head -n ${ientry} | tail -n 1)
               # get peer process
               peer=$(../../../tools/tracedump ${vftr_binary}_${irank}.vfd | \
                      awk '$2=="recv" && $3!="end"{getline;print;}' | \
                      sed 's/=/ /g' | \
                      sort -nk 9 | \
                      awk '{print $9}' | \
                      head -n ${ientry} | tail -n 1)
               # Check if actually used message size is consistent
               # with expected message size
               if [[ "${count}" -ne "${nb}" ]] ; then
                  echo "Message recv size from rank ${jrank} by ${irank} is ${count}!"
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
      else
         jrank=0
         for irun in $(seq 1 1 ${nr});
         do
            ientry=$(bc <<< "${irun}")
            # Validate sending
            # Get actually used message size
            count=$(../../../tools/tracedump ${vftr_binary}_${irank}.vfd | \
                    awk '$2=="send" && $3!="end"{getline;print;}' | \
                    sed 's/=/ /g' | \
                    sort -nk 9 | \
                    awk '{print $2}' | \
                    head -n ${ientry} | tail -n 1)
            # get peer process
            peer=$(../../../tools/tracedump ${vftr_binary}_${irank}.vfd | \
                   awk '$2=="send" && $3!="end"{getline;print;}' | \
                   sed 's/=/ /g' | \
                   sort -nk 9 | \
                   awk '{print $9}' | \
                   head -n ${ientry} | tail -n 1)
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
               echo "Message send from rank ${irank} by ${peer}!"
               echo "Was expecting sending to rank ${jrank}!"
               exit 1;
            fi
         done
      fi
   done
done
