#!/bin/bash
set -x
export VFTR_PROF_TRUNCATE=no

test_name=no_instrument_attribute

if [ "x$HAS_MPI" == "xYES" ]; then
   ${MPI_EXEC} ${MPI_OPTS} ${NP} 1 ./${test_name} || exit 1
else
   ./${test_name} || exit 1
fi

ncalls=$(grep " not_instrumented_function" ${test_name}_0.log | wc -l)
if [ "${ncalls}" -gt "0" ]; then
   echo "The function `not_instrumented_function` should not appear in the profile"
   exit 1;
fi
ncalls=$(grep " instrumented_function" ${test_name}_0.log | wc -l)
if [ "${ncalls}" -eq "0" ]; then
   echo "The function `instrumented_function` should appear in the profile"
   exit 1
fi
exit 0
