#!/bin/bash

source ${srcdir}/../environment/filenames.sh

set -x
vftr_binary=hwprof_region3
nprocs=1

determine_bin_prefix $vftr_binary

logfile=$(get_logfile_name ${vftr_binary} "all")

configfile=${vftr_binary}.json
cat << EOF > ${configfile}
{
   "mpi": {"show_table": false},
   "hwprof": {
      "show_counters": true,
      "type": "dummy",
      "counters": [
          {
             "hwc_name": "FOO",
             "symbol": "f"
          }
      ],
      "observables": [
          {
             "name": "test",
             "formula": "f"
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
value=${value%%.*} # Strip decimal places
target=10
if [ "${value}" != "${target}" ]; then
   echo "user-region-1 should have observable value ${target} in ${logfile}."
   echo "Found ${value}"
   exit 1;
fi
