#!/bin/bash
set -x
test_name=symbols_1
input_file=${srcdir}/ref_input/symbols_1.x
output_file=symbols_1.out
ref_file=${srcdir}/ref_output/symbols_1.out

if [ "x$HAS_MPI" == "xYES" ]; then
   ${MPI_EXEC} ${MPI_OPTS} ${NP} 1 ./${test_name} ${input_file} > ${output_file}
else
   ./${test_name} ${input_file} > ${output_file}
fi

if [ "$?" == "0" ]; then
  diff ${output_file} ${ref_file}
else
  exit 1
fi
