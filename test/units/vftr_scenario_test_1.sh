#!/bin/bash
set -x
ref_out_dir=ref_output
ref_in_dir=ref_input
testname=vftr_scenario_test_1
outfile=$testname.out

rm -f $outfile

if [ "x$HAS_MPI" == "xYES" ]; then
   $MPI_EXEC $NP 1 ./test_vftrace $testname $ref_in_dir/$testname.json
else
  ./test_vftrace $testname $ref_in_dir/$testname.json
fi

last_success=$?
if [ $last_success == 0 ]; then
  diff $ref_out_dir/$outfile $outfile
else
  exit  $last_success
fi

