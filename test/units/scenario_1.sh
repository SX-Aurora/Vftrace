#!/bin/bash
set -x
test_name=scenario_1
input_file=ref_input/$test_name.json
ref_file=ref_output/$test_name.out
output_file=$test_name.out

rm -f $outfile

if [ "x$HAS_MPI" == "xYES" ]; then
   ${MPI_EXEC} ${MPI_OPTS} ${NP} ${nprocs} 1 ./$test_name $input_file > $output_file
else
  ./$test_name $input_file > $output_file
fi

last_success=$?
if [ $last_success == 0 ]; then
  diff $ref_file $output_file
else
  exit  $last_success
fi

