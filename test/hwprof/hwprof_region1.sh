#!/bin/bash

source ${srcdir}/../environment/filenames.sh

set -x
vftr_binary=hwprof_region1
nprocs=1

determine_bin_prefix $vftr_binary

logfile=$(get_logfile_name ${vftr_binary} "all")

configfile=${vftr_binary}.json
cat << EOF > ${configfile}
{
   "mpi": {"show_table": false},
   "hwprof": {
      "observables": [
          {
             "name": "test",
             "formula": "1"
          }
      ]
   }
}
EOF
export VFTR_CONFIG=${configfile}

if [ "x${HAS_MPI}" == "xYES" ]; then
   ${MPI_EXEC} ${MPI_OPTS} ${NP} ${nprocs} ./${vftr_binary} || exit 1
else
   ./${vftr_binary} || exit 1
fi

value=`grep -m 2 user-region-1 $logfile | tail -1 | awk '{print $6}'`
target=1.000000
if [ "${value}" != "${target}" ]; then
   echo "user-region-1 should have observable value ${target} in ${logfile}."
   echo "Found ${value}"
   exit 1;
fi
