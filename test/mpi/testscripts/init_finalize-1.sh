#!/bin/bash

vftr_binary=init_finalize_1
configfile=${vftr_binary}.json
nprocs=4
echo ${MPI_EXEC}
echo ${MPI_OPTS}
echo ${NP}
echo ${nprocs}
${MPI_EXEC} ${MPI_OPTS} ${NP} ${nprocs} ./${vftr_binary} || exit 1
