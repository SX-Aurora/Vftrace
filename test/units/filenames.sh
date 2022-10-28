#!/bin/bash

set -x
test_name=filenames
configfile=${test_name}.json
outfile=${test_name}.out
ref_file=${srcdir}/ref_output/${outfile}

# create logfile
echo "{\"outfile_basename\": \"${test_name}\"}" > ${configfile}
export VFTR_CONFIG=${configfile}

rm -f ${outfile}

if [ "x${HAS_MPI}" == "xYES" ]; then
   ${MPI_EXEC} ${MPI_OPTS} ${NP} 1 ./${test_name} > ${outfile} || exit 1
else
   ./${test_name} > ${outfile} || exit 1
fi

diff ${ref_file} ${outfile}
