#!/bin/bash
set -x
test_name=stacktree_parallel_1
output_file=${test_name}.out
ref_file=${srcdir}/ref_output/${test_name}.out

rm -f ${output_file}

${MPI_EXEC} ${MPI_OPTS} ${NP} 2 ./${test_name} > ${output_file}

last_success=$?

# We need to filter out the Address lines because they are
# not reproducible. 
# diff file1 <(expression) does not work when called with make check
if [ $last_success == 0 ]; then
  diff ${output_file} ${ref_file}
else
  exit  ${last_success}
fi

