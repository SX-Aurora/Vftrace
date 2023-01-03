#!/bin/bash

source ${srcdir}/../../environment/filenames.sh

set -x

vftr_binary=init_finalize_3
configfile=${vftr_binary}.json
nprocs=4

determine_bin_prefix $vftr_binary
logfile=$(get_logfile_name $vftr_binary 0)

echo "{\"logfile_for_ranks\": \"all\", \"sampling\": {\"active\": true}}" > ${configfile}
export VFTR_CONFIG=${configfile}

${MPI_EXEC} ${MPI_OPTS} ${NP} ${nprocs} ./${vftr_binary} || exit 1

cat ${logfile}

n=$(cat ${logfile} | \
    grep -i "MPI_Init<\|MPI_Init_f08<" | wc -l)
if [[ ${n} -le 0 ]] ; then
   exit 1;
fi

n=$(cat ${logfile} | \
    grep -i "MPI_Finalize<\|MPI_Finalize_f08<" | wc -l)
if [[ ${n} -le 0 ]] ; then
   exit 1;
fi
