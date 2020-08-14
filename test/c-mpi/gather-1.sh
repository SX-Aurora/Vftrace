#!/bin/bash

vftr_binary=gather
nprocs=6
nb=4

export VFTR_SAMPLING="Yes"
export VFTR_MPI_LOG="Yes"
export VFTR_PRECISE="MPI_*"

mpirun -np ${nprocs} ./${vftr_binary} ${nb} || exit 1
