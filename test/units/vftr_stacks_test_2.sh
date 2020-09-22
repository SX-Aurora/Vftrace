#!/bin/bash
set -x
ref_out_dir=ref_output
ref_in_dir=ref_input
testname=vftr_stacks_test_2
outfile=$testname.out

rm -f $outfile

if [ "x$HAS_MPI" == "xYES" ]; then
   $MPI_EXEC $NP 4 ./test_vftrace $testname
else
   ./test_vftrace $testname
fi

last_success=$?
if [ $last_success == 0 ]; then
  # There is one temporary output file for each rank.
  # We just put one after the other.
  cat ${outfile}_0 ${outfile}_1 ${outfile}_2 ${outfile}_3  > $outfile
  diff $ref_out_dir/$outfile $outfile
else
  exit  $last_success
fi


