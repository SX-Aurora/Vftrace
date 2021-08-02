#!/bin/bash
set -x
ref_out_dir=${srcdir}/ref_output
vftr_binary=setup_1
outfile=setup_1.out

rm -f $outfile

if [ "x$HAS_MPI" == "xYES" ]; then
   ${MPI_EXEC} ${MPI_OPTS} ${NP} 1 ./${vftr_binary} > $outfile
else
   ./${vftr_binary} > $outfile
fi

last_success=$?
if [ $last_success == 0 ]; then
  diff $ref_out_dir/$outfile $outfile
else
  exit  $last_success
fi
