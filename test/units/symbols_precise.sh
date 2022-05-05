#!/bin/bash
set -x
test_name=symbols_precise

if [ "x$HAS_MPI" == "xYES" ]; then
   ${MPI_EXEC} ${MPI_OPTS} ${NP} 1 ./${test_name} ${input_file} || exit 1
else
   ./${test_name} ${input_file}
fi
