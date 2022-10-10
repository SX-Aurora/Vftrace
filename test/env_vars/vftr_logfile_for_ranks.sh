#!/bin/bash

source ${srcdir}/../environment/filenames.sh

set -x
test_name=vftr_logfile_for_ranks
output_file=vftr_logfile_for_ranks.out
ref_file=${srcdir}/ref_output/little_tasks.out
nranks=1
if [ "x${HAS_MPI}" == "xYES" ]; then
   nranks=4
fi

function run_binary() {
   if [ "x${HAS_MPI}" == "xYES" ]; then
      ${MPI_EXEC} ${MPI_OPTS} ${NP} ${nranks} ./${test_name} > ${output_file} || exit 1
   else
      ./${test_name} > ${output_file} || exit 1
   fi
}

rm_outfiles $output_file "" $test_name
run_binary
diff ${output_file} ${ref_file} || exit 1
check_file_exists $(get_logfile_name $test_name "all")
for irank in $(seq 0 1 $(bc <<< "${nranks}-1"));
do
   check_file_notexists $(get_logfile_name $test_name ${irank})
done

export VFTR_LOGFILE_FOR_RANKS="none"
rm_outfiles $output_file "" $test_name
run_binary
diff ${output_file} ${ref_file} || exit 1
check_file_exists $(get_logfile_name $test_name "all")
for irank in $(seq 0 1 $(bc <<< "${nranks}-1"));
do
   check_file_notexists $(get_logfile_name $test_name ${irank})
done

export VFTR_LOGFILE_FOR_RANKS="all"
rm_outfiles $output_file "" $test_name
run_binary
diff ${output_file} ${ref_file} || exit 1
check_file_exists $(get_logfile_name $test_name "all")
for irank in $(seq 0 1 $(bc <<< "${nranks}-1"));
do
   check_file_exists $(get_logfile_name $test_name ${irank})
done

if [ "x${HAS_MPI}" == "xYES" ]; then
   export VFTR_LOGFILE_FOR_RANKS="1,3"
   rm_outfiles $output_file "" $test_name
   run_binary
   diff ${output_file} ${ref_file} || exit 1
   check_file_exists $(get_logfile_name $test_name "all")
   check_file_notexists $(get_logfile_name $test_name 0)
   check_file_exists $(get_logfile_name $test_name 1)
   check_file_notexists $(get_logfile_name $test_name 2)
   check_file_exists $(get_logfile_name $test_name 3)

   export VFTR_LOGFILE_FOR_RANKS="0-2"
   rm_outfiles $output_file "" $test_name
   run_binary
   diff ${output_file} ${ref_file} || exit 1
   check_file_exists $(get_logfile_name $test_name "all")
   check_file_exists $(get_logfile_name $test_name 0)
   check_file_exists $(get_logfile_name $test_name 1)
   check_file_exists $(get_logfile_name $test_name 2)
   check_file_notexists $(get_logfile_name $test_name 3)

   export VFTR_LOGFILE_FOR_RANKS="0,2-3"
   rm_outfiles $output_file "" $test_name
   run_binary
   diff ${output_file} ${ref_file} || exit 1
   check_file_exists $(get_logfile_name $test_name "all")
   check_file_exists $(get_logfile_name $test_name 0)
   check_file_notexists $(get_logfile_name $test_name 1)
   check_file_exists $(get_logfile_name $test_name 2)
   check_file_exists $(get_logfile_name $test_name 3)
fi
