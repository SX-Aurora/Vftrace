#!/bin/bash
set -x
test_name=collatehashes_parallel
output_file=${test_name}.out
ref_file=${srcdir}/ref_output/${test_name}.out

rm -f ${output_file}
rm ${test_name}_p*.tmpout

if [ "x$HAS_MPI" == "xYES" ]; then
   ${MPI_EXEC} ${MPI_OPTS} ${NP} 2 ./${test_name} || exit 1
else
   ./${test_name} || exit 1
fi

last_success=$?

cat ${test_name}_p0.tmpout \
    ${test_name}_p1.tmpout > ${output_file}

if [ $last_success == 0 ]; then
  diff ${output_file} ${ref_file}
else
  exit  ${last_success}
fi

