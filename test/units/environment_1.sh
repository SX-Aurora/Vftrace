#!/bin/bash
set -x
vftr_binary=environment_1

output_file=environment_1.out
ref_file=ref_output/$output_file

if [ "x$HAS_MPI" == "xYES" ]; then
   ${MPI_EXEC} ${MPI_OPTS} ${NP} 1 ./${vftr_binary} ${output_file}
else
   ./${vftr_binary} ${output_file}
fi

if [ "$?" == "0" ]; then
  diff ${output_file} ${ref_file}
else
  exit 1
fi
