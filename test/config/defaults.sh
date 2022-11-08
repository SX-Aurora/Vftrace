#!/bin/bash
set -x
test_name=defaults
outfile=defaults.out
ref_file=${srcdir}/ref_output/${outfile}

rm -f ${outfile}

if [ "x${HAS_MPI}" == "xYES" ]; then
   ${MPI_EXEC} ${MPI_OPTS} ${NP} 1 ./${test_name} > ${outfile} || exit 1
else
   ./${test_name} > ${outfile} || exit 1
fi

diff ${outfile} ${ref_file} || exit 1
