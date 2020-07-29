#!/bin/bash
set -x
ref_out_dir=ref_output
ref_in_dir=ref_input
testname=vftr_environment_test_1
outfile=$testname.out

rm -f $outfile

# export VFTR_OFF=yes

./test_vftrace $testname

