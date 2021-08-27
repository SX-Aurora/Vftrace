#!/bin/bash
set -x
export VFTR_PROF_TRUNCATE=no

test_name=no_instrument_attribute_C

if [ "x$HAS_MPI" == "xYES" ]; then
   ${MPI_EXEC} ${MPI_OPTS} ${NP} 1 ./${test_name} || exit 1
else
   ./${test_name} || exit 1
fi

ncalls=$(grep not_instrumented_function ${test_name}_0.log | wc -l)
if [ "${ncalls}" -gt "0" ]; then
   exit 1;
else
   exit 0
fi
