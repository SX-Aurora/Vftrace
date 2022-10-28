#!/bin/bash

set -x
test_name=print_name_grouped_prof_table
configfile=${test_name}.json
output_file=${test_name}.out
ref_file=${srcdir}/ref_output/${test_name}.out

# create logfile
echo "{\"name_grouped_profile_table\": {\"sort_table\": {\"column\": \"time_excl\", \"ascending\": false}}}" > ${configfile}
export VFTR_CONFIG=${configfile}

rm -f ${output_file}

if [ "x${HAS_MPI}" == "xYES" ]; then
   ${MPI_EXEC} ${MPI_OPTS} ${NP} 1 ./${test_name} > ${output_file} || exit 1
else
   ./${test_name} > ${output_file} || exit 1
fi

diff ${output_file} ${ref_file}
