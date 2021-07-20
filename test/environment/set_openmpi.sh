#!/bin/bash
export HAS_MPI="YES"
export MPI_EXEC="mpirun"
export NP="-np"
# Some tests require four mpi-processes
# On Systems with fewer cores a special oversubscribe
# flag is required in order to run properly.
# With OpenMPI version 5 the old flag is deprecated
if [ "x$OMPI_VERSION_LT5" == "xYES" ]; then
   export MPI_OPTS="--map-by :OVERSUBSCRIBE"
else
   export MPI_OPTS="--oversubscribe"
fi
