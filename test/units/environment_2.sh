#!/bin/bash
set -x
vftr_binary=environment_2
outfile=environment_2.out
ref_file=${srcdir}/ref_output/$output_file

rm -f $outfile

if [ "x$HAS_MPI" == "xYES" ]; then
  ${MPI_EXEC} ${MPI_OPTS} ${NP} 1 ./${vftr_binary} > $outfile
else
  ./${vftr_binary} > $outfile
fi

last_success=$?
echo "last_success: $last_success"

if [ $last_success == 0 ]; then
  diff $ref_file $outfile
else
  exit $last_success
fi
