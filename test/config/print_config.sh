#!/bin/bash

source ${srcdir}/../environment/filenames.sh

set -x
test_name=print_config
output_file=${test_name}.out
configfile=${test_name}.json
ref_file=${srcdir}/ref_output/little_tasks.out
logfile=${test_name}_all.log

#if [ "${IS_SHARED_BUILD}" == "YES" ]; then
#   logfile=lt-$logfile
#   vfdfile=lt-$vfdfile
#fi

determine_bin_prefix ${test_name}
logfile=${BIN_PREFIX}${logfile}
vfdfile=${BIN_PREFIX}${vfdfile}


function run_binary () {
   rm_outfiles ${output_file} "" ${test_name}
   if [ "x${HAS_MPI}" == "xYES" ]; then
      ${MPI_EXEC} ${MPI_OPTS} ${NP} 1 ./${test_name} > ${output_file} || exit 1
   else
      ./${test_name} > ${output_file} || exit 1
   fi
   diff ${output_file} ${ref_file} || exit 1
}

run_binary
check_file_exists $logfile
diff ${output_file} ${ref_file} || exit 1
count=$(grep "Vftrace default configuration:" ${logfile} | wc -l)
if [[ "${count}" -ne "1" ]] ; then
   echo "Expected printed configuration"
   exit 1
fi

export VFTR_CONFIG="${configfile}"
echo "{\"print_config\": false}" > ${configfile}
run_binary
check_file_exists $logfile
diff ${output_file} ${ref_file} || exit 1
count=$(grep "Vftrace configuration read from" ${logfile} | wc -l)
if [[ "${count}" -ne "0" ]] ; then
   echo "Expected no printed configuration"
   exit 1
fi

echo "{\"print_config\": true}" > ${configfile}
export VFTR_CONFIG="${configfile}"
run_binary
diff ${output_file} ${ref_file} || exit 1
check_file_exists $logfile
count=$(grep "Vftrace configuration read from" ${logfile} | wc -l)
if [[ "${count}" -ne "1" ]] ; then
   echo "Expected printed configuration"
   exit 1
fi
