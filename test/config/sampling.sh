#!/bin/bash

source ${srcdir}/../environment/filenames.sh

set -x
test_name=sampling
configfile=${test_name}.json
output_file=${test_name}.out
ref_file=${srcdir}/ref_output/little_tasks.out

determine_bin_prefix $test_name

logfile=$(get_logfile_name $test_name "all")
vfdfile=$(get_vfdfile_name $test_name "0")

rm_outfiles $output_file "" $test_name
if [ "x${HAS_MPI}" == "xYES" ]; then
   ${MPI_EXEC} ${MPI_OPTS} ${NP} 1 ./${test_name} > ${output_file} || exit 1
else
   ./${test_name} > ${output_file} || exit 1
fi
diff ${output_file} ${ref_file} || exit 1
check_file_exists $logfile
check_file_notexists $vfdfile

export VFTR_CONFIG=${configfile}
echo "{\"sampling\": {\"active\": true}}" > ${configfile}

rm_outfiles $output_file "" $test_name
if [ "x${HAS_MPI}" == "xYES" ]; then
   ${MPI_EXEC} ${MPI_OPTS} ${NP} 1 ./${test_name} > ${output_file} || exit 1
else
   ./${test_name} > ${output_file} || exit 1
fi
diff ${output_file} ${ref_file} || exit 1
check_file_exists $logfile
check_file_exists $vfdfile
