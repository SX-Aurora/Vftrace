#!/bin/bash
set -x
test_name=defaults2
outfile=${test_name}.out
configfile=${test_name}.json
ref_file=${srcdir}/ref_output/little_tasks.out

../../tools/config_tools/vftrace_generate_default_config > ${configfile}
export VFTR_CONFIG=${configfile}

if [ "x${HAS_MPI}" == "xYES" ]; then
   ${MPI_EXEC} ${MPI_OPTS} ${NP} 1 ./${test_name} > ${outfile} || exit 1
else
   ./${test_name} > ${outfile} || exit 1
fi

diff ${outfile} ${ref_file} || exit 1
