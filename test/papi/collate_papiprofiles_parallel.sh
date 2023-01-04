#!/bin/bash
set -x
test_name=collate_papiprofiles_parallel
exe_name=collate_papiprofiles_parallel
output_file=${test_name}.out
ref_file=${srcdir}/ref_output/${output_file}

rm -f ${output_file}
${MPI_EXEC} ${MPI_OPTS} ${NP} 2 ./${exe_name} > ${output_file} || exit 1

diff ${output_file} ${ref_file}
