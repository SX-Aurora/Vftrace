#!/bin/bash
set -x
test_name=scenario_3
input_file=ref_input/$test_name.json
output_file=$test_name.out

rm -f $output_file

if [ "x$HAS_MPI" == "xYES" ]; then
   ${MPI_EXEC} ${MPI_OPTS} ${NP} 1 ./test_name $input_file > $output_file || exit 1
else
  ./$test_name $input_file || exit 1
fi


# This is supposed to fail and is not diffed to reference output,
# so there is no test for the return value here
