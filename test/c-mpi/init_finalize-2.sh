#!/bin/bash

vftr_binary=init_finalize
nprocs=2

export VFTR_SAMPLING="Yes"
export VFTR_MPI_LOG="Yes"
export VFTR_PRECISE="MPI_*"

mpirun -np ${nprocs} ./${vftr_binary}
