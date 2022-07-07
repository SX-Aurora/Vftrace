#!/bin/bash
set -x
vftr_binary=parse_and_print
outfile=advisor.out
ref_file=${srcdir}/ref_output/$output_file

rm -f $outfile

export VFTR_OF=yes # Should be VFTR_OFF
export VFTR_SMPLING=yes # Should be VFTR_SAMPLING

if [ "x$HAS_MPI" == "xYES" ]; then
  ${MPI_EXEC} ${MPI_OPTS} ${NP} 1 ./${vftr_binary} > $outfile || exit 1
else
  ./${vftr_binary} > $outfile || exit 1
fi

diff $ref_file $outfile
