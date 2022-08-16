#!/bin/bash
set -x
test_name=collatempiprofiles
output_file=${test_name}.out
ref_file=${srcdir}/ref_output/${test_name}.out

rm -f ${output_file}

export VFTR_MPI_LOG="yes"
${MPI_EXEC} ${MPI_OPTS} ${NP} 1 ./${test_name} > ${output_file} || exit 1

diff ${output_file} ${ref_file}
