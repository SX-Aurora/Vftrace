#!/bin/bash

vftr_binary=init_thread_finalize_2
nprocs=4

export VFTR_SAMPLING="Yes"
export VFTR_MPI_LOG="Yes"

${MPI_EXEC} ${MPI_OPTS} ${NP} ${nprocs} ./${vftr_binary} || exit 1
