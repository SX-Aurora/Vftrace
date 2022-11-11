#!/bin/bash

set -x

vftr_binary=no_init_finalize
configfile=${vftr_binary}.json

./${vftr_binary} || exit 1

cat ${vftr_binary}_all.log

n=$(cat ${vftr_binary}_all.log | \
    grep -i "MPI_Init<\|MPI_Init_f08<" | wc -l)
if [[ ${n} -gt 0 ]] ; then
   exit 1;
fi

n=$(cat ${vftr_binary}_all.log | \
    grep -i "MPI_Finalize<\|MPI_Finalize_f08<" | wc -l)
if [[ ${n} -gt 0 ]] ; then
   exit 1;
fi
