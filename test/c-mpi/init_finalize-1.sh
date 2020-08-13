#!/bin/bash

vftr_binary=init_finalize
nprocs=4

mpirun -np ${nprocs} ./${vftr_binary} || exit 1
