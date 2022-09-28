#!/bin/bash

source ${srcdir}/../environment/filenames.sh

set -x
test_name=vftr_logfile_basename
output_file=vftr_logfile_basename.out
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
export VFTR_LOGFILE_BASENAME=${testbasename}
logfile="${testbasename}_all.log"

rm_outfiles $output_file "" $test_name

if [ "x${HAS_MPI}" == "xYES" ]; then
   ${MPI_EXEC} ${MPI_OPTS} ${NP} 1 ./${test_name} > ${output_file} || exit 1
else
   ./${test_name} > ${output_file} || exit 1
fi

diff ${output_file} ${ref_file} || exit 1

check_file_exists $logfile

