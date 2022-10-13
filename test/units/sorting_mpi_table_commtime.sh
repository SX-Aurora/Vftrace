#!/bin/bash
set -x
test_name=sorting_mpi_table
output_file=${test_name}_commtime.out
ref_file=${srcdir}/ref_output/${test_name}_commtime.out

rm -f ${output_file}

export VFTR_SORT_MPI_TABLE="COMM_TIME"
if [ "x${HAS_MPI}" == "xYES" ]; then
   ${MPI_EXEC} ${MPI_OPTS} ${NP} 1 ./${test_name} > ${output_file} || exit 1
else
   ./${test_name} > ${output_file} || exit 1
fi

diff ${output_file} ${ref_file}
