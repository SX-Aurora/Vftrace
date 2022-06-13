#!/bin/bash
vftr_binary=radixsort_double

listsize=$(bc <<< "32*${RANDOM}+128")
if [ "x$HAS_MPI" == "xYES" ]; then
   ${MPI_EXEC} ${MPI_OPTS} ${NP} 1 ./${vftr_binary} ${listsize} || exit 1
else
   ./${vftr_binary} ${listsize} || exit 1
fi
