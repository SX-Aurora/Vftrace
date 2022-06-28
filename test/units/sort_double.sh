#!/bin/bash
vftr_binary=sort_double

listsize=$(bc <<< "32*${RANDOM}+128")
echo "ascending:"
if [ "x$HAS_MPI" == "xYES" ]; then
   ${MPI_EXEC} ${MPI_OPTS} ${NP} 1 ./${vftr_binary} ${listsize} 1 || exit 1
else
   ./${vftr_binary} ${listsize} 1 || exit 1
fi

echo "descending:"
if [ "x$HAS_MPI" == "xYES" ]; then
   ${MPI_EXEC} ${MPI_OPTS} ${NP} 1 ./${vftr_binary} ${listsize} 0 || exit 1
else
   ./${vftr_binary} ${listsize} 0 || exit 1
fi
