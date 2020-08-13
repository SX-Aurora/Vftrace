#!/bin/bash

vftr_binary=init_finalize
nprocs=2

mpirun -np ${nprocs} ./${vftr_binary} || exit 1
