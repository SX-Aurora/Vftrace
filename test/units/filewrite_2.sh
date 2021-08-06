#!/bin/bash
set -x
vftr_binary=filewrite_2
outfile=filewrite_2.out
if [ "x${HAS_MPI}" == "xYES" ] ; then
   ref_file=${srcdir}/ref_output/mpi/${outfile}
else
   ref_file=${srcdir}/ref_output/serial/$outfile
fi

rm -f $outfile

if [ "x$HAS_MPI" == "xYES" ]; then
   ${MPI_EXEC} ${MPI_OPTS} ${NP} 1 ./${vftr_binary} > $outfile
else
   ./${vftr_binary} > $outfile
fi

last_success=$?
if [ $last_success == 0 ]; then
  diff $ref_file $outfile
else
  exit  $last_success
fi

