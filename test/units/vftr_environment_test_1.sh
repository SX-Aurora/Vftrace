#!/bin/bash
set -x
ref_out_dir=${srcdir}/ref_output
ref_in_dir=${srcdir}/ref_input
testname=vftr_environment_test_1
outfile=$testname.out

if [ -f $outfile ] ; then
   rm -f $outfile
fi

# The user might have set some VFTR_ environment variables.
# We save them in an array and unset them all.
# After the test, we reset them to their original value.
vftr_variables=(`env | grep VFTR_`)
for v in ${vftr_variables[@]}; do
  unset `echo $v | cut -f1 -d "="`
done
./test_vftrace $testname
diff $ref_out_dir/$outfile $outfile

# Reset all environment variables here
for v in ${vftr_variables[@]}; do
  export $v;
done
