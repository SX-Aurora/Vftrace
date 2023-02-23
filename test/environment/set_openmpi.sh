#!/bin/bash
export HAS_MPI="YES"
export MPI_EXEC="mpirun"
export NP="-np"
# Some tests require four mpi-processes
# On Systems with fewer cores a special oversubscribe
# flag is required in order to run properly.
# With OpenMPI version 5 the old flag is deprecated
if [ "x${OMPI_VERSION_LT5}" == "xYES" ]; then
   export MPI_OPTS="--oversubscribe"
else
   export MPI_OPTS="--map-by :OVERSUBSCRIBE"
fi

# Some systems have been observed to show this error message:
#
# "A high-performance Open MPI point-to-point messaging module
#  was unable to find any relevant network interfaces"
#  [...]
# Another transport will be used instead, although this may result in
# lower performance.
# 
# This is an issue to be resolved within the MPI environment. As the
# message indicates, this issue is not relevant for the functionality
# tests, since it only affects the choice of the interconnect. However,
# the warning message appears in stderr. Tests which redirect stderr
# and compare it to a reference file will therefore fail. The following
# environment variable switches off the warning.
export OMPI_MCA_btl_base_warn_component_unused=0

