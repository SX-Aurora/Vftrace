#!/bin/bash
export HAS_MPI="YES"
export MPI_EXEC="mpirun"
export NP="-np"
if [ "x$OMPI_VERSION_LT5" == "xYES" ]; then
   export MPI_OPTS="--map-by :OVERSUBSCRIBE"
else
   export MPI_OPTS="--oversubscribe"
fi
