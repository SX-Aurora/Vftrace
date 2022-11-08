#!/bin/bash
set -x
test_name=collatempiprofiles
configfile=${test_name}.json
output_file=${test_name}.out
ref_file=${srcdir}/ref_output/${test_name}.out

rm -f ${output_file}

echo "{\"mpi\": {\"show_table\": true, \"log_messages\": true}}" > ${configfile}
export VFTR_CONFIG=${configfile}
${MPI_EXEC} ${MPI_OPTS} ${NP} 1 ./${test_name} > ${output_file} || exit 1

diff ${output_file} ${ref_file}
