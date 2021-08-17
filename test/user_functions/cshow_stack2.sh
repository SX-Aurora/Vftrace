#!/bin/bash

vftr_binary=cshow_stack2
nprocs=1

if [ "x$HAS_MPI" == "xYES" ]; then
   ${MPI_EXEC} ${MPI_OPTS} ${NP} ${nprocs} ./${vftr_binary} || exit 1
else
   ./${vftr_binary} || exit 1
fi

grep "Stack trees traced by user: 2" ${vftr_binary}_0.log
if [ $? -ne "0" ]; then
  exit 1;
fi
grep "2: func1<main<init" ${vftr_binary}_0.log
if [ $? -ne "0" ]; then
  exit 1;
fi
grep "3: func2<main<init" ${vftr_binary}_0.log
if [ $? -ne "0" ]; then
  exit 1;
fi
