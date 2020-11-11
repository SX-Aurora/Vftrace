#!/bin/bash
set -x
ref_out_dir=ref_output
ref_in_dir=ref_input
testname=vftr_filewrite_test_2
outfile=$testname.out

rm -f $outfile

if [ "x$HAS_MPI" == "xYES" ]; then
   ${MPI_EXEC} ${MPI_OPTS} ${NP} 1 ./test_vftrace $testname
else
   ./test_vftrace $testname
fi

last_success=$?

# In case of an MPI build, MPI_Init is included
# in the function list, and measures actual system
# times, not dummies. These system times are not
# reproducable, and we cut them out.
if [ $last_success == 0 ]; then
  grep --invert-match MPI_Init $outfile | diff $ref_out_dir/$outfile -
else
  exit $last_success
fi
