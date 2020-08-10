#!/bin/bash
set -x
ref_out_dir=${srcdir}/ref_output
ref_in_dir=${srcdir}/ref_input
testname=vftr_symbols_test_1
outfile=$testname.out

if [ -f $outfile ] ; then
   rm -f $outfile
fi

./test_vftrace $testname $ref_in_dir/$testname.x
diff $ref_out_dir/$outfile $outfile
