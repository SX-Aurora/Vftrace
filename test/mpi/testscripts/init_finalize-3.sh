#!/bin/bash

set -x

vftr_binary=init_finalize_3
configfile=${vftr_binary}.json
nprocs=4

echo "{\"logfile_for_ranks\": \"all\", \"sampling\": {\"active\": true}}" > ${configfile}
export VFTR_CONFIG=${configfile}

${MPI_EXEC} ${MPI_OPTS} ${NP} ${nprocs} ./${vftr_binary} || exit 1

cat ${vftr_binary}_0.log

n=$(cat ${vftr_binary}_0.log | \
    grep -i "MPI_Init<\|MPI_Init_f08<" | wc -l)
if [[ ${n} -le 0 ]] ; then
   exit 1;
fi

n=$(cat ${vftr_binary}_0.log | \
    grep -i "MPI_Finalize<\|MPI_Finalize_f08<" | wc -l)
if [[ ${n} -le 0 ]] ; then
   exit 1;
fi
