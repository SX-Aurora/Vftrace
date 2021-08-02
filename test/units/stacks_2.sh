#!/bin/bash
set -x
test_name=stacks_2
output_file=$test_name.out
if [ "x$HAS_MPI" == "xYES" ]; then
   ref_file=${srcdir}/ref_output/mpi/$test_name.out
fi

rm -f $output_file

if [ "x$HAS_MPI" == "xYES" ]; then
   ${MPI_EXEC} ${MPI_OPTS} ${NP} 4 ./${test_name} > $output_file
fi

last_success=$?
if [ $last_success == 0 ]; then
  # There is one temporary output file for each rank.
  # We just put one after the other.
  # cat ${output_file}_0 ${output_file}_1 ${output_file}_2 ${output_file}_3  > $output_file
  diff $ref_file $output_file
else
  exit  $last_success
fi


