#!/bin/bash

vftr_binary=pcontrol
nprocs=2
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
   for irank in $(seq 0 1 $(bc <<< "${nprocs}-1"));
   do
      ../../../tools/vftrace_vfd_dump ${vftr_binary}_${irank}.vfd
      if [ "${irank}" -eq "0" ] ; then
         jrank=1
         ipeer=$(bc <<< "${jrank}")
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
      elif [ "${irank}" -eq "1" ] ; then
         # Validate receiving
         # Check if calls to Pcontrol were executed
         n=$(../../../tools/vftrace_vfd_dump ${vftr_binary}_${irank}.vfd | \
             grep -i "call MPI_Pcontrol" | wc -l)
         if [[ ${n} -le 0 ]] ; then
            echo "Number of MPI_Pcontrol calls (${n}) does not match expected value (0)"
            exit 1;
         fi
         
         n=$(../../../tools/vftrace_vfd_dump ${vftr_binary}_${irank}.vfd | \
             grep -i "exit MPI_Pcontrol" | wc -l)
         if [[ ${n} -le 0 ]] ; then
            echo "Number of MPI_Pcontrol exits (${n}) does not match expected value (0)"
            exit 1;
         fi
         # MPI-logging should be disabled due to Pcontrol(0)
         # The instrumentation should still be effective
         n=$(../../../tools/vftrace_vfd_dump ${vftr_binary}_${irank}.vfd | \
             grep -i "call MPI_Recv" | wc -l)
         if [[ ${n} -le 0 ]] ; then
            echo "Number of MPI_Recv calls (${n}) does not match expected value (0)"
            exit 1;
         fi
         
         n=$(../../../tools/vftrace_vfd_dump ${vftr_binary}_${irank}.vfd | \
             grep -i "exit MPI_Recv" | wc -l)
         if [[ ${n} -le 0 ]] ; then
            echo "Number of MPI_Recv exits (${n}) does not match expected value (0)"
            exit 1;
         fi
         # Count the amount of logged receives
         n=$(../../../tools/vftrace_vfd_dump ${vftr_binary}_${irank}.vfd | \
             awk '$2=="recv" && $3!="end"{getline;print;}' | \
             wc -l)
         if [[ ${n} -gt 0 ]] ; then
            echo "Number of logged receives (${n}) does not match expected value (0)"
            exit 1;
         fi
      fi
   done
done
