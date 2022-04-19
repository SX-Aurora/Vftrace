#!/bin/bash
set -x
test_name=collatestacks_parallel_3
output_file=${test_name}.out
ref_file=${srcdir}/ref_output/${test_name}.out

rm -f ${output_file}
rm ${test_name}_p*.tmpout

${MPI_EXEC} ${MPI_OPTS} ${NP} 2 ./${test_name} || exit 1

cat ${test_name}_p0.tmpout \
    ${test_name}_p1.tmpout > ${output_file}

diff ${output_file} ${ref_file}
