#!/bin/bash
set -x
test_name=collatehashes_parallel
output_file=${test_name}.out
ref_file=${srcdir}/ref_output/${test_name}.out

rm -f ${output_file}

if [ "x$HAS_MPI" == "xYES" ]; then
   ${MPI_EXEC} ${MPI_OPTS} ${NP} 2 ./${test_name} > ${output_file}
else
   ./${test_name} > ${output_file}
fi

last_success=$?

if [ $last_success == 0 ]; then
  diff ${output_file} ${ref_file}
else
  exit  ${last_success}
fi

