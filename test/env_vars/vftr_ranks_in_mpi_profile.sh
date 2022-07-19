#!/bin/bash
set -x
test_name=vftr_ranks_in_mpi_profile
output_file=${test_name}.out
ref_file=${srcdir}/ref_output/mpi_tasks.out
nranks=4

function run_binary() {
   if [ "x$HAS_MPI" == "xYES" ]; then
      ${MPI_EXEC} ${MPI_OPTS} ${NP} ${nranks} ./${test_name} \
         > ${output_file} || exit 1
   else
      ./${test_name} > ${output_file} || exit 1
   fi
}

function rm_outfiles() {
   for file in ${output_file} ${test_name}_*.log ${test_name}_*.vfd;
   do
      if [ -f ${file} ] ; then
         rm ${file}
      fi
   done
}

function check_mpi_entry_exists() {
   for irank in $@;
   do
      logfile="${test_name}_${irank}.log"
      nmpientries=$(cat ${logfile} | \
                    grep -A4 "Communication profile" | \
                    grep -i MPI_Alltoallv | wc -l)
      if [ "${nmpientries}" -ne "1" ] ; then
         echo "expected one entry of MPI_Alltoallv in Communication profile " \
              "of rank ${irank}, but found ${nmpientries}!"
         exit 1
      fi
   done
}

function check_mpi_entry_notexists() {
   for irank in $@;
   do
      logfile="${test_name}_${irank}.log"
      nmpientries=$(cat ${logfile} | \
                    grep -A4 "Communication profile" | \
                    grep -i MPI_Alltoallv | wc -l)
      if [ "${nmpientries}" -ne "0" ] ; then
         echo "expected no entry of MPI_Alltoallv in Communication profile " \
              "of rank ${irank}, but found ${nmpientries}!"
         exit 1
      fi
   done
}

function check_mpi_communication_consistency() {
   for irank in $@;
   do 
      logfile="${test_name}_${irank}.log"

      # compute reference values
      nsendmsg_ref=0
      nrecvmsg_ref=0
      nsendbyteavg_ref=0 
      nrecvbyteavg_ref=0 
      for jrank in $@;
      do 
         nsendmsg_ref=$(bc <<< "${nsendmsg_ref} + 1")
         nrecvmsg_ref=$(bc <<< "${nrecvmsg_ref} + 1")
         nsendbyteavg_ref=$(bc <<< "${nsendbyteavg_ref}+2^${irank}")
         nrecvbyteavg_ref=$(bc <<< "${nrecvbyteavg_ref}+2^${jrank}")
      done
      nsendbyteavg_ref=$(bc -l <<< "${nsendbyteavg_ref}*4/${nsendmsg_ref}")
      nsendbyteavg_ref=$(printf "%.f" ${nsendbyteavg_ref})
      nrecvbyteavg_ref=$(bc -l <<< "${nrecvbyteavg_ref}*4/${nrecvmsg_ref}")
      nrecvbyteavg_ref=$(printf "%.f" ${nrecvbyteavg_ref})
      nmsg_ref=$(bc <<< "${nsendmsg_ref} + ${nrecvmsg_ref}")

      # get logged values
      profentry=$(cat ${logfile} | \
                  grep -A4 "Communication profile" | \
                  grep -i MPI_Alltoallv)
      nmsg=$(echo ${profentry} | awk '{print $2}')
      nsendbyteavg=$(printf "%.f" $(echo ${profentry} | awk '{print $4}'))
      nrecvbyteavg=$(printf "%.f" $(echo ${profentry} | awk '{print $6}'))

      if [ "${nmsg}" -ne "${nmsg_ref}" ] ; then
         echo "Expected ${nmsg_ref} messages in communication profile " \
              "for rank ${irank}, but found ${nmsg}!"
         exit 1
      fi
      if [ "${nsendbyteavg}" -ne "${nsendbyteavg_ref}" ] ; then
         echo "Expected ${nsendbyteavg_ref} send bytes on average " \
              "in communication profile for rank ${irank}, but found ${nsendbyteavg}!"
         exit 1
      fi
      if [ "${nrecvbyteavg}" -ne "${nrecvbyteavg_ref}" ] ; then
         echo "Expected ${nrecvbyteavg_ref} recv bytes on average " \
              "in communication profile for rank ${irank}, but found ${nrecvbyteavg}!"
         exit 1
      fi
   done
}

if [ "x$HAS_MPI" == "xYES" ]; then
   export VFTR_MPI_LOG="yes"
   export VFTR_LOGFILE_FOR_RANKS="all"

   rm_outfiles
   run_binary
   diff ${output_file} ${ref_file} || exit 1
   check_mpi_entry_exists 0 1 2 3
   check_mpi_communication_consistency 0 1 2 3

   export VFTR_RANKS_IN_MPI_PROFILE="0,1"
   rm_outfiles
   run_binary
   diff ${output_file} ${ref_file} || exit 1
   check_mpi_entry_exists 0 1
   check_mpi_entry_notexists 2 3
   check_mpi_communication_consistency 0 1

   export VFTR_RANKS_IN_MPI_PROFILE="1-3"
   rm_outfiles
   run_binary
   diff ${output_file} ${ref_file} || exit 1
   check_mpi_entry_exists 1 2 3
   check_mpi_entry_notexists 0
   check_mpi_communication_consistency 1 2 3
fi
