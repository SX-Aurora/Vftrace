#!/bin/bash
set -x
test_name=sorting_mpi_table
configfile=${test_name}_sendsize.json
output_file=${test_name}_sendsize.out
ref_file=${srcdir}/ref_output/${test_name}_sendsize.out

rm -f ${output_file}

echo "{\"mpi\": {\"sort_table\": {\"column\": \"send_size\", \"ascending\":false}}}" > ${configfile}
export VFTR_CONFIG=${configfile}
if [ "x${HAS_MPI}" == "xYES" ]; then
   ${MPI_EXEC} ${MPI_OPTS} ${NP} 1 ./${test_name} > ${output_file} || exit 1
else
   ./${test_name} > ${output_file} || exit 1
fi

diff ${output_file} ${ref_file}
