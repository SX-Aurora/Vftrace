#!/bin/bash

set -x
test_name=sorting_stacks
configfile=${test_name}_stackid.json
output_file=${test_name}_stackid.out
ref_file=${srcdir}/ref_output/${test_name}_stackid.out

# create logfile
echo "{\"profile_table\": {\"sort_table\": {\"column\": \"stack_id\", \"ascending\": true}}}" > ${configfile}
export VFTR_CONFIG=${configfile}

rm -f ${output_file}

if [ "x${HAS_MPI}" == "xYES" ]; then
   ${MPI_EXEC} ${MPI_OPTS} ${NP} 1 ./${test_name} > ${output_file} || exit 1
else
   ./${test_name} > ${output_file} || exit 1
fi

diff ${output_file} ${ref_file}
