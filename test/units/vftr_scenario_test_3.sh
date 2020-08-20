#!/bin/bash
set -x
ref_out_dir=ref_output
ref_in_dir=ref_input
testname=vftr_scenario_test_3
outfile=$testname.out

rm -f $outfile
if [ "x$HAS_MPI" == "xYES" ]; then
   $MPI_EXEC $NP 1 ./test_vftrace $testname $ref_in_dir/$testname.json
else
  ./test_vftrace $testname $ref_in_dir/$testname.json
fi


# This is supposed to fail and is not diffed to reference output,
# so there is no test for the return value here
