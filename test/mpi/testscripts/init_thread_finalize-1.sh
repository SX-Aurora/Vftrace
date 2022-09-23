#!/bin/bash

vftr_binary=init_thread_finalize_1
nprocs=4
echo ${MPI_EXEC}
echo ${MPI_OPTS}
echo ${NP}
echo ${nprocs}
${MPI_EXEC} ${MPI_OPTS} ${NP} ${nprocs} ./${vftr_binary} || exit 1