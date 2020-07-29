#!/bin/bash
set -x
ref_out_dir=ref_output
ref_in_dir=ref_input
testname=vftr_symbols_test_1
outfile=$testname.out

rm -f $outfile
./test_vftrace $testname $ref_in_dir/$testname.x
diff $ref_out_dir/$outfile $outfile
