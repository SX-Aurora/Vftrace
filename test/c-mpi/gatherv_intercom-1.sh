#!/bin/bash

vftr_binary=gatherv_intercom
nprocs=6
nb=1024

export VFTR_SAMPLING="Yes"
export VFTR_MPI_LOG="Yes"
export VFTR_PRECISE="MPI_*"

mpirun -np ${nprocs} ./${vftr_binary} ${nb} || exit 1
