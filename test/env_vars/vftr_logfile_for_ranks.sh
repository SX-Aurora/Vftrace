#!/bin/bash
set -x
test_name=vftr_logfile_for_ranks
output_file=vftr_logfile_for_ranks.out
ref_file=${srcdir}/ref_output/little_tasks.out
nranks=1
if [ "x${HAS_MPI}" == "xYES" ]; then
   nranks=4
fi

function rm_outfiles() {
   for file in ${output_file} ${test_name}_*.log ${test_name}_*.vfd;
   do
      if [ -f ${file} ] ; then
         rm ${file}
      fi
   done
}

function run_binary() {
   if [ "x${HAS_MPI}" == "xYES" ]; then
      ${MPI_EXEC} ${MPI_OPTS} ${NP} ${nranks} ./${test_name} > ${output_file} || exit 1
   else
      ./${test_name} > ${output_file} || exit 1
   fi
}

function check_logfile_exists() {
   logfile="${test_name}_$1.log"
   if [ ! -f ${logfile} ] ; then
      echo "Could not find logfile \"${logfile}\"!"
      exit 1
   fi
}

function check_logfile_notexists() {
   logfile="${test_name}_$1.log"
   if [ -f ${logfile} ] ; then
      echo "Logfile \"${logfile}\" does exist although it should not!"
      exit 1
   fi
}

rm_outfiles
run_binary
diff ${output_file} ${ref_file} || exit 1
check_logfile_exists "all"
for irank in $(seq 0 1 $(bc <<< "${nranks}-1"));
do
   check_logfile_notexists "${irank}"
done

export VFTR_LOGFILE_FOR_RANKS="none"
rm_outfiles
run_binary
diff ${output_file} ${ref_file} || exit 1
check_logfile_exists "all"
for irank in $(seq 0 1 $(bc <<< "${nranks}-1"));
do
   check_logfile_notexists "${irank}"
done

export VFTR_LOGFILE_FOR_RANKS="all"
rm_outfiles
run_binary
diff ${output_file} ${ref_file} || exit 1
check_logfile_exists "all"
for irank in $(seq 0 1 $(bc <<< "${nranks}-1"));
do
   check_logfile_exists "${irank}"
done

if [ "x${HAS_MPI}" == "xYES" ]; then
   export VFTR_LOGFILE_FOR_RANKS="1,3"
   rm_outfiles
   run_binary
   diff ${output_file} ${ref_file} || exit 1
   check_logfile_exists "all"
   check_logfile_notexists "0"
   check_logfile_exists "1"
   check_logfile_notexists "2"
   check_logfile_exists "3"

   export VFTR_LOGFILE_FOR_RANKS="0-2"
   rm_outfiles
   run_binary
   diff ${output_file} ${ref_file} || exit 1
   check_logfile_exists "all"
   check_logfile_exists "0"
   check_logfile_exists "1"
   check_logfile_exists "2"
   check_logfile_notexists "3"

   export VFTR_LOGFILE_FOR_RANKS="0,2-3"
   rm_outfiles
   run_binary
   diff ${output_file} ${ref_file} || exit 1
   check_logfile_exists "all"
   check_logfile_exists "0"
   check_logfile_notexists "1"
   check_logfile_exists "2"
   check_logfile_exists "3"
fi
