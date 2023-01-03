#!/bin/bash

source ${srcdir}/../../environment/filenames.sh

vftr_binary=probe
configfile=${vftr_binary}.json
nprocs=2

echo "{\"sampling\": {\"active\": true}}" > ${configfile}
export VFTR_CONFIG=${configfile}

nb=$(bc <<< "32*${RANDOM}")

${MPI_EXEC} ${MPI_OPTS} ${NP} ${nprocs} ./${vftr_binary} ${nb} || exit 1

irank=1

determine_bin_prefix $vftr_binary
vfdfile=$(get_vfdfile_name ${vftr_binary} ${irank})
../../../tools/vftrace_vfd_dump ${vfdfile}

n=$(../../../tools/vftrace_vfd_dump ${vfdfile} | \
    grep -i "call MPI_Probe" | wc -l)
if [[ ${n} -le 0 ]] ; then
   exit 1;
fi

n=$(../../../tools/vftrace_vfd_dump ${vfdfile} | \
    grep -i "exit MPI_Probe" | wc -l)
if [[ ${n} -le 0 ]] ; then
   exit 1;
fi
