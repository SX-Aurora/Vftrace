#!/bin/bash
set -x
test_name=environment_1
output_file=$test_name.out
ref_file=${srcdir}/ref_output/$output_file

rm -f $outfile

if [ "x$HAS_MPI" == "xYES" ]; then
   ${MPI_EXEC} ${MPI_OPTS} ${NP} 1 ./${test_name} > ${output_file}
else
   ./${test_name} > ${output_file}
fi

if [ "$?" == "0" ]; then
  diff ${output_file} ${ref_file}
else
  exit 1
fi
