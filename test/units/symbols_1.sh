#!/bin/bash
set -x
test_name=symbols_1

if [ "x$HAS_MPI" == "xYES" ]; then
   ${MPI_EXEC} ${MPI_OPTS} ${NP} 1 ./${test_name} || exit 1
else
   ./${test_name}
fi
