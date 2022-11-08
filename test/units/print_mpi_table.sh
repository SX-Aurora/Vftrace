#!/bin/bash
set -x
test_name=print_mpi_table
configfile=${test_name}.json
output_file=${test_name}.out
ref_file=${srcdir}/ref_output/${test_name}.out

rm -f ${output_file}

echo "{\"mpi\": {\"sort_table\": {\"column\": \"none\", \"ascending\":true}}}" > ${configfile}
export VFTR_CONFIG=${configfile}
if [ "x${HAS_MPI}" == "xYES" ]; then
   ${MPI_EXEC} ${MPI_OPTS} ${NP} 1 ./${test_name} > ${output_file} || exit 1
else
   ./${test_name} > ${output_file} || exit 1
fi

diff ${output_file} ${ref_file}
