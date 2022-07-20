#!/bin/bash
set -x
vftr_binary=advisor2
outfile=${vftr_binary}.out
errfile=${vftr_binary}.err
ref_outfile=${srcdir}/ref_output/little_tasks.out
ref_errfile=${srcdir}/ref_output/advisor.out

rm -f ${outfile} ${errfile} ${errfile}_sorted

export VFTR_OF=yes # Should be VFTR_OFF
export VFTR_SMPLING=yes # Should be VFTR_SAMPLING

if [ "x$HAS_MPI" == "xYES" ]; then
  ${MPI_EXEC} ${MPI_OPTS} ${NP} 1 ./${vftr_binary} > $outfile 2> $errfile || exit 1
else
  ./${vftr_binary} > $outfile 2> $errfile || exit 1
fi


cat ${errfile} | sort > ${errfile}_sorted
diff $ref_outfile $outfile || exit 1
diff $ref_errfile ${errfile}_sorted || exit 1
