#!/bin/bash
set -x
ref_out_dir=ref_output
ref_in_dir=ref_input
testname=vftr_functions_test_2
outfile=$testname.out

rm -f $outfile

if [ "x$HAS_MPI" == "xYES" ]; then
   $MPI_EXEC $NP 1 ./test_vftrace $testname
else
   ./test_vftrace $testname
fi

last_success=$?
# We need to filter out the Address lines because they are
# not reproducible. 
# diff file1 <(expression) does not work when called with make check
if [ $last_success == 0 ]; then
  grep --invert-match Address $outfile  | diff $ref_out_dir/$outfile -
else
  exit  $last_success
fi

