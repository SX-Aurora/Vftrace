#!/bin/bash
set -x
vftr_binary=symbols_test_1

input_file=ref_input/symbols_test_1.x
output_file=symbols_test_1.out
ref_file=ref_output/symbols_test_1.out
if [ "x$HAS_MPI" == "xYES" ]; then
   ${MPI_EXEC} ${MPI_OPTS} ${NP} 1 ./${vftr_binary} ${input_file} ${output_file}
else
   ./${vftr_binary} ${input_file} ${output_file}
fi

if [ "$?" == "0" ]; then
  diff ${output_file} ${ref_file}
else
  exit 1
fi
