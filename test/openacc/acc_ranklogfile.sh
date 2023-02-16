#!/bin/bash
set -x

nvidia-smi
has_gpu=`echo $?`

source ${srcdir}/../environment/filenames.sh

test_name=acc_ranklogfile
vftr_binary=acc_region2
configfile=${test_name}.json

determine_bin_prefix $vftr_binary

logfile=$(get_logfile_name ${vftr_binary} "all")

cat << EOF > ${configfile}
{
   "logfile_for_ranks": "0",
   "openacc": {
        "show_event_details": true
    }
}
EOF
export VFTR_CONFIG=${configfile}

if [ "x${HAS_MPI}" == "xYES" ]; then
   ${MPI_EXEC} ${MPI_OPTS} ${NP} ${nprocs} ./${vftr_binary} || exit 1
else
   ./${vftr_binary} || exit 1
fi

if [ $has_gpu -eq 0 ]; then # On GPU
  ## Search for the wait events in the detailled event table
  nwait=`grep "wait | 0x" ${logfile} | wc -l`
  if [ "${nwait}" -ne 2 ]; then
     echo "wait event not found exactly 2 times."
     exit 1;
  fi
else # On Host
  grep "No OpenACC events have been registered" ${logfile}
  exit $?
fi
