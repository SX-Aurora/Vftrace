#!/bin/bash

vftr_binary=cshow_stack2
nprocs=1
export VFTR_LOGFILE_BASENAME=${vftr_binary}
logfile=${vftr_binary}_0.log

if [ "x$HAS_MPI" == "xYES" ]; then
   ${MPI_EXEC} ${MPI_OPTS} ${NP} ${nprocs} ./${vftr_binary} || exit 1
else
   ./${vftr_binary} || exit 1
fi

cat $logfile
echo "***********"

grepfor="Stack trees traced by user: 2"
grep "${grepfor}" $logfile
if [ $? -ne "0" ]; then
  echo "Fail: String $grepfor not found!"
  exit 1;
fi

grepfor="func1<main<init"
nfound=`grep "${grepfor}" $logfile | wc -l`
if [ $nfound -ne "2" ]; then
  echo "Fail: String $grepfor not found two times but $nfound"
  exit 1;
fi

grepfor="func2<main<init"
nfound=`grep "${grepfor}" $logfile | wc -l`
if [ $nfound -ne "2" ]; then
  echo "Fail: String $grepfor not found two times but $nfound"
  exit 1;
fi
