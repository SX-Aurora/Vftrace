#!/bin/bash
set -x
test_name=environment_1
output_file=$test_name.out
ref_file=ref_output/$output_file

rm -f $outfile

# The user might have set some VFTR_ environment variables.
# We save them in an array and unset them all.
# After the test, we reset them to their original value.
vftr_variables=(`env | grep VFTR_`)
for v in ${vftr_variables[@]}; do
  unset `echo $v | cut -f1 -d "="`
done


if [ "x$HAS_MPI" == "xYES" ]; then
   ${MPI_EXEC} ${MPI_OPTS} ${NP} 1 ./${test_name} > ${output_file}
else
   ./${test_name} > ${output_file}
fi

# Reset all environment variables here
for v in ${vftr_variables[@]}; do
  export $v;
done

if [ "$?" == "0" ]; then
  diff ${output_file} ${ref_file}
else
  exit 1
fi
