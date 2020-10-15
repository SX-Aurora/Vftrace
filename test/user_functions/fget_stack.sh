#!/bin/bash

vftr_binary=fget_stack
nprocs=1

export VFTR_SAMPLING="Yes"
export VFTR_PROF_TRUNCATE="no"
export VFTR_PRECISE="fkt*"

tmpfile=$(mktemp)

if [ "x$HAS_MPI" == "xYES" ]; then
   ${MPI_EXEC} ${NP} ${nprocs} ./${vftr_binary} ${maxnreg} > ${tmpfile} || exit 1
else
   ./${vftr_binary} ${maxnreg} > ${tmpfile} || exit 1
fi

callstack=$(cat ${tmpfile})
rm ${tmpfile}

refstack=""
for i in $(seq 1 1 3);
do
   refstack="fkt${i}<${refstack}"
done

echo "Callstack: ${callstack}"
echo "Expected stack: ${refstack}"

hit=$(echo ${callstack} | grep "${refstack}" | wc -l)

if [ "${hit}" -ne "1" ] ; then
   echo "Callstack \"${callstack}\" does not match expected callstack \"${refstack}\""
   exit 1;
fi
