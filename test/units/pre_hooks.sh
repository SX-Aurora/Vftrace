#!/bin/bash

set -x

test_name=pre_hooks
logfile=${test_name}_all.log
output_file=${test_name}.out
ref_file=${srcdir}/ref_output/${test_name}.out

rm -f ${output_file}

if [ "x${HAS_MPI}" == "xYES" ]; then
   ${MPI_EXEC} ${MPI_OPTS} ${NP} 1 ./${test_name} > ${output_file} || exit 1
else
   ./${test_name} > ${output_file} || exit 1
fi

diff ${output_file} ${ref_file} || exit 1

cat ${logfile}
# Count how often the before_main function appears in the profile
count=$(cat ${logfile} | grep "before_main")
if [[ "${count}" -ne "0" ]] ; then
   echo "Expected not to find \"before_main\" in logfile, but found it ${count} times."
   exit 1;
fi
