#!/bin/bash

vftr_binary=init_finalize_2
nprocs=4

export VFTR_SAMPLING="Yes"
export VFTR_MPI_LOG="Yes"
export VFTR_PRECISE="MPI_*|mpi_*"

mpirun -np ${nprocs} ./${vftr_binary} || exit 1
