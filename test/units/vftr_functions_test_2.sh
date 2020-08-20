#!/bin/bash
set -x
ref_out_dir=ref_output
ref_in_dir=ref_input
testname=vftr_functions_test_2
outfile=$testname.out

rm -f $outfile

./test_vftrace $testname
# We need to filter out the Address lines because they are
# not reproducible. 
# diff file1 <(expression) does not work when called with make check
grep --invert-match Address $outfile  | diff $ref_out_dir/$outfile -
