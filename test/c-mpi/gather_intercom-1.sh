#!/bin/bash

vftr_binary=gather_intercom
nprocs=4
nb=1024

export VFTR_SAMPLING="Yes"
export VFTR_MPI_LOG="Yes"
export VFTR_PRECISE="MPI_*"

mpirun -np ${nprocs} ./${vftr_binary} ${nb} || exit 1
