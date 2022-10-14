#!/bin/bash
vftr_binary=search_collated_stacks_name

if [ "x${HAS_MPI}" == "xYES" ]; then
   ${MPI_EXEC} ${MPI_OPTS} ${NP} 1 ./${vftr_binary} ${listsize} 1 || exit 1
else
   ./${vftr_binary} ${listsize} 1 || exit 1
fi
