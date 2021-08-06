#!/bin/bash
set -x
test_name=functions_1
output_file=$test_name.out
if [ "x$HAS_MPI" == "xYES" ]; then
   ref_file=${srcdir}/ref_output/mpi/$test_name.out
else
   ref_file=${srcdir}/ref_output/serial/$test_name.out
fi

rm -f $output_file

if [ "x$HAS_MPI" == "xYES" ]; then
   ${MPI_EXEC} ${MPI_OPTS} ${NP} 1 ./${test_name} > ${output_file}
else
   ./${test_name} > ${output_file}
fi

last_success=$?

# We need to filter out the Address lines because they are
# not reproducible. 
# diff file1 <(expression) does not work when called with make check
if [ $last_success == 0 ]; then
  grep --invert-match Address $output_file  | diff $ref_file -
else
  exit  $last_success
fi

