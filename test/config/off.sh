#!/bin/bash

source ${srcdir}/../environment/filenames.sh

set -x
test_name=off
output_file=${test_name}.out
configfile=${test_name}.json
ref_file=${srcdir}/ref_output/little_tasks.out
logfile=${test_name}_all.log
vfdfile=${test_name}_0.vfd

function run_binary () {
   rm_outfiles ${output_file} "" ${test_name}
   if [ "x${HAS_MPI}" == "xYES" ]; then
      ${MPI_EXEC} ${MPI_OPTS} ${NP} 1 ./${test_name} > ${output_file} || exit 1
   else
      ./${test_name} > ${output_file} || exit 1
   fi
   diff ${output_file} ${ref_file} || exit 1
}
export VFTR_CONFIG="${configfile}"
echo "{\"sampling\": {\"active\": true}}" > ${configfile}

# Test with environment variable
run_binary
check_file_exists $logfile
check_file_exists $vfdfile

export VFTR_OFF="no"
run_binary
check_file_exists $logfile
check_file_exists $vfdfile

export VFTR_OFF="yes"
run_binary
check_file_notexists $logfile
check_file_notexists $vfdfile

unset VFTR_OFF

# Test with config file
run_binary
check_file_exists $logfile
check_file_exists $vfdfile

echo "{\"off\": false, \"sampling\": {\"active\": true}}" > ${configfile}
run_binary
check_file_exists $logfile
check_file_exists $vfdfile

echo "{\"off\": true, \"sampling\": {\"active\": true}}" > ${configfile}
export VFTR_CONFIG="${configfile}"
run_binary
check_file_notexists $logfile
check_file_notexists $vfdfile
