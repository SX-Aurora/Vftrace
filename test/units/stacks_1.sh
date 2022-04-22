#!/bin/bash
set -x
test_name=stacks_1
output_file=$test_name.out
if [ "x$HAS_MPI" == "xYES" ]; then
   ref_file=${srcdir}/ref_output/mpi/$test_name.out
else
   ref_file=${srcdir}/ref_output/serial/$test_name.out
fi

rm -f $output_file

if [ "x$HAS_MPI" == "xYES" ]; then
   ${MPI_EXEC} ${MPI_OPTS} ${NP} 1 ./${test_name} > $output_file || exit 1
else
   ./${test_name} > $output_file || exit 1
fi

diff $ref_file $output_file
