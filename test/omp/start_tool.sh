#!/bin/bash
vftr_binary=start_tool

export OMP_NUM_THREADS=2
if [ "x$HAS_MPI" == "xYES" ]; then
   ${MPI_EXEC} ${MPI_OPTS} ${NP} 1 ./${vftr_binary} || exit 1
else
   ./${vftr_binary} || exit 1
fi
