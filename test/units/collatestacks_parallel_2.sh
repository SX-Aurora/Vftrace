#!/bin/bash
set -x
test_name=collatestacks_parallel_2
output_file=${test_name}.out
ref_file=${srcdir}/ref_output/${test_name}.out

rm -f ${output_file}

${MPI_EXEC} ${MPI_OPTS} ${NP} 1 ./${test_name}_1 : \
            ${MPI_OPTS} ${NP} 1 ./${test_name}_2 > ${output_file} || exit 1

diff ${output_file} ${ref_file}

