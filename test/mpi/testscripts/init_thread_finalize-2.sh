#!/bin/bash

vftr_binary=init_thread_finalize_2
configfile=${vftr_binary}.json
nprocs=4

echo "{\"sampling\": {\"active\": true}, \"mpi\": {\"active\": true}}" > ${configfile}
export VFTR_CONFIG=${configfile}

${MPI_EXEC} ${MPI_OPTS} ${NP} ${nprocs} ./${vftr_binary} || exit 1
