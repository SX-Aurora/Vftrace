#!/bin/bash
set -x
test_name=scenario_1
input_file=${srcdir}/ref_input/${test_name}.json
ref_file=${srcdir}/ref_output/${test_name}.out
output_file=${test_name}.out

rm -f ${output_file}

if [ "x${HAS_MPI}" == "xYES" ]; then
   ${MPI_EXEC} ${MPI_OPTS} ${NP} 1 ./${test_name} ${input_file} > ${output_file} || exit 1
else
  ./${test_name} ${input_file} > ${output_file} || exit 1
fi

diff ${ref_file} ${output_file}
