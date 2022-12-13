#!/bin/bash

source ${srcdir}/../environment/filenames.sh

set -x
test_name=logfile_for_ranks
configfile=${test_name}.json
output_file=${test_name}.out
ref_file=${srcdir}/ref_output/little_tasks.out
nranks=1
if [ "x${HAS_MPI}" == "xYES" ]; then
   nranks=4
fi

determine_bin_prefix $test_name

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

export VFTR_CONFIG=${configfile}
echo "{\"logfile_for_ranks\": \"none\"}" > ${configfile}
rm_outfiles $output_file "" $test_name
run_binary
diff ${output_file} ${ref_file} || exit 1
check_file_exists $(get_logfile_name $test_name "all")
for irank in $(seq 0 1 $(bc <<< "${nranks}-1"));
do
   check_file_notexists $(get_logfile_name $test_name ${irank})
done

echo "{\"logfile_for_ranks\": \"all\"}" > ${configfile}
export VFTR_CONFIG=${configfile}
rm_outfiles $output_file "" $test_name
run_binary
diff ${output_file} ${ref_file} || exit 1
check_file_exists $(get_logfile_name $test_name "all")
for irank in $(seq 0 1 $(bc <<< "${nranks}-1"));
do
   check_file_exists $(get_logfile_name $test_name ${irank})
done

if [ "x${HAS_MPI}" == "xYES" ]; then
   echo "{\"logfile_for_ranks\": \"0\"}" > ${configfile}
   export VFTR_CONFIG=${configfile}
   rm_outfiles $output_file "" $test_name
   run_binary
   diff ${output_file} ${ref_file} || exit 1
   check_file_exists $(get_logfile_name $test_name "all")
   check_file_exists $(get_logfile_name $test_name 0)
   check_file_notexists $(get_logfile_name $test_name 1)
   check_file_notexists $(get_logfile_name $test_name 2)
   check_file_notexists $(get_logfile_name $test_name 3)

   echo "{\"logfile_for_ranks\": \"1,3\"}" > ${configfile}
   export VFTR_CONFIG=${configfile}
   rm_outfiles $output_file "" $test_name
   run_binary
   diff ${output_file} ${ref_file} || exit 1
   check_file_exists $(get_logfile_name $test_name "all")
   check_file_notexists $(get_logfile_name $test_name 0)
   check_file_exists $(get_logfile_name $test_name 1)
   check_file_notexists $(get_logfile_name $test_name 2)
   check_file_exists $(get_logfile_name $test_name 3)

   echo "{\"logfile_for_ranks\": \"0-2\"}" > ${configfile}
   export VFTR_CONFIG=${configfile}
   rm_outfiles $output_file "" $test_name
   run_binary
   diff ${output_file} ${ref_file} || exit 1
   check_file_exists $(get_logfile_name $test_name "all")
   check_file_exists $(get_logfile_name $test_name 0)
   check_file_exists $(get_logfile_name $test_name 1)
   check_file_exists $(get_logfile_name $test_name 2)
   check_file_notexists $(get_logfile_name $test_name 3)

   echo "{\"logfile_for_ranks\": \"0,2-3\"}" > ${configfile}
   export VFTR_CONFIG=${configfile}
   rm_outfiles $output_file "" $test_name
   run_binary
   diff ${output_file} ${ref_file} || exit 1
   check_file_exists $(get_logfile_name $test_name "all")
   check_file_exists $(get_logfile_name $test_name 0)
   check_file_notexists $(get_logfile_name $test_name 1)
   check_file_exists $(get_logfile_name $test_name 2)
   check_file_exists $(get_logfile_name $test_name 3)
fi
