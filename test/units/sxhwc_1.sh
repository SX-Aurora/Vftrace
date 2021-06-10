#!/bin/bash
set -x
test_name=sxhwc_1
output_file=$test_name.out
ref_file=ref_output/$output_file

rm -f $output_file

if [ "x$HAS_MPI" == "xYES" ]; then
   ${MPI_EXEC} ${MPI_OPTS} ${NP} 1 ./${test_name} > $output_file
else
   ./${test_name} > $output_file
fi

last_success=$?
if [ $last_success == 0 ]; then
  diff $ref_file $output_file
else
  exit  $last_success
fi


