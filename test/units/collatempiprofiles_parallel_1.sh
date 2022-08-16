#!/bin/bash
set -x
test_name=collatempiprofiles_parallel_1
output_file=${test_name}.out
ref_file=${srcdir}/ref_output/${test_name}.out

rm -f ${output_file}

${MPI_EXEC} ${MPI_OPTS} ${NP} 2 ./${test_name} > ${output_file} || exit 1

diff ${output_file} ${ref_file}
