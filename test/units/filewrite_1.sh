#!/bin/bash

set -x
vftr_binary=filewrite_1
export VFTR_LOGFILE_BASENAME=$vftr_binary
outfile=filewrite_1.out
ref_file=${srcdir}/ref_output/$outfile

rm -f $outfile

if [ "x$HAS_MPI" == "xYES" ]; then
   ${MPI_EXEC} ${MPI_OPTS} ${NP} 1 ./${vftr_binary} > $outfile || exit 1
else
   ./${vftr_binary} > $outfile || exit 1
fi

diff $ref_file $outfile
