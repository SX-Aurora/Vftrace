#!/bin/bash

vftr_binary=init_finalize_1
nprocs=4

mpirun -np ${nprocs} ./${vftr_binary} || exit 1
