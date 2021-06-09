#!/bin/bash
set -x
vftr_binary=environment_2
outfile=environment_2.out
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
  ${MPI_EXEC} ${MPI_OPTS} ${NP} 1 ./${vftr_binary} > $outfile
else
  ./${vftr_binary} > $outfile
fi

last_success=$?
echo "last_success: $last_success"

# Reset all environment variables here
for v in ${vftr_variables[@]}; do
  export $v;
done

if [ $last_success == 0 ]; then
  diff $ref_file $outfile
else
  exit $last_success
fi
