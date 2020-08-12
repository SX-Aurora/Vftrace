#!/bin/bash

vftr_binary=wait
nprocs=2
nb=1024

export VFTR_SAMPLING="Yes"
export VFTR_MPI_LOG="Yes"
export VFTR_PRECISE="MPI_*"

mpirun -np ${nprocs} ./${vftr_binary} ${nb}
