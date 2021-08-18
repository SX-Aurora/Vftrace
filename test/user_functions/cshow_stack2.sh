#!/bin/bash

vftr_binary=cshow_stack2
nprocs=1

if [ "x$HAS_MPI" == "xYES" ]; then
   ${MPI_EXEC} ${MPI_OPTS} ${NP} ${nprocs} ./${vftr_binary} || exit 1
else
   ./${vftr_binary} || exit 1
fi

cat ${vftr_binary}_0.log
echo "***********"

grepfor="Stack trees traced by user: 2"
grep "${grepfor}" ${vftr_binary}_0.log
if [ $? -ne "0" ]; then
  echo "Fail: String $grepfor not found!"
  exit 1;
fi

grepfor="func1<main<init"
nfound=`grep "${grepfor}" ${vftr_binary}_0.log | wc -l`
if [ $nfound -ne "2" ]; then
  echo "Fail: String $grepfor not found two times but $nfound"
  exit 1;
fi

grepfor="func2<main<init"
nfound=`grep "${grepfor}" ${vftr_binary}_0.log | wc -l`
if [ $nfound -ne "2" ]; then
  echo "Fail: String $grepfor not found two times but $nfound"
  exit 1;
fi
