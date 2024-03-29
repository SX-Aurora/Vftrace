#!/bin/bash

vftr_binary=init_finalize_2
configfile=${vftr_binary}.json
nprocs=4

echo "{\"sampling\": {\"active\": true}}" > ${configfile}
export VFTR_CONFIG=${configfile}

${MPI_EXEC} ${MPI_OPTS} ${NP} ${nprocs} ./${vftr_binary} || exit 1
