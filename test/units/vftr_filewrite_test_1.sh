#!/bin/bash
set -x
ref_out_dir=ref_output
ref_in_dir=ref_input
testname=vftr_filewrite_test_1
outfile=$testname.out

rm -f $outfile
echo $HAS_MPI

if [ "x$HAS_MPI" == "xYES" ]; then
   $MPI_EXEC $NP 1 ./test_vftrace $testname
else
   ./test_vftrace $testname
fi
diff $ref_out_dir/$outfile $outfile

