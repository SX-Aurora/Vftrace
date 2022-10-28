#!/bin/bash

source ${srcdir}/../environment/filenames.sh

set -x
test_name=outfile_basename
configfile=${test_name}.json
output_file=${test_name}.out
ref_file=${srcdir}/ref_output/little_tasks.out
logfile=$(get_logfile_name ${test_name} "all")

rm_outfiles $output_file "" $test_name
if [ "x${HAS_MPI}" == "xYES" ]; then
   ${MPI_EXEC} ${MPI_OPTS} ${NP} 1 ./${test_name} > ${output_file} || exit 1
else
   ./${test_name} > ${output_file} || exit 1
fi
diff ${output_file} ${ref_file} || exit 1
check_file_exists $logfile

testbasename="TestBasename"
echo "{\"outfile_basename\": \"${testbasename}\"}" > ${configfile}
export VFTR_CONFIG=${configfile}
logfile="${testbasename}_all.log"

rm_outfiles $output_file "" $test_name
if [ "x${HAS_MPI}" == "xYES" ]; then
   ${MPI_EXEC} ${MPI_OPTS} ${NP} 1 ./${test_name} > ${output_file} || exit 1
else
   ./${test_name} > ${output_file} || exit 1
fi
diff ${output_file} ${ref_file} || exit 1

check_file_exists $logfile
