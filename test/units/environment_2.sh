#!/bin/bash
set -x
vftr_binary=environment_2
outfile=environment_2.out
ref_file=${srcdir}/ref_output/$output_file

rm -f $outfile

if [ "x$HAS_MPI" == "xYES" ]; then
  ${MPI_EXEC} ${MPI_OPTS} ${NP} 1 ./${vftr_binary} > $outfile || exit 1
else
  ./${vftr_binary} > $outfile || exit 1
fi

diff $ref_file $outfile
