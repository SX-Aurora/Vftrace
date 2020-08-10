#!/bin/bash
set -x
ref_out_dir=${srcdir}/ref_output
ref_in_dir=${srcdir}/ref_input
testname=vftr_filewrite_test_1
outfile=$testname.out

if [ -f $outfile ] ; then
   rm -f $outfile
fi

./test_vftrace $testname
diff $ref_out_dir/$outfile $outfile

