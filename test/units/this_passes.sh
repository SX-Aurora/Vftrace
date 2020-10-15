#!/bin/bash
set -x

if [ "x$HAS_MPI" == "xYES" ]; then
  $MPI_EXEC $NP 1 ./test_vftrace this_passes
else
  ./test_vftrace this_passes
fi
