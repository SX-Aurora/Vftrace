#!/bin/bash
set -x

nvidia-smi
has_gpu=`echo $?`

source ${srcdir}/../environment/filenames.sh

test_name=collate_acc_parallel1
vftr_binary=${test_name}
configfile=${vftr_binary}.json

determine_bin_prefix ${vftr_binary}

logfile_all=$(get_logfile_name ${vftr_binary} "all")
logfile_0=$(get_logfile_name ${vftr_binary} "0")
logfile_1=$(get_logfile_name ${vftr_binary} "1")

export VFTR_CONFIG=${configfile}
cat << EOF > ${configfile}
{
  "logfile_for_ranks": "all",
  "openacc": {
     "active": true
  }
}
EOF

if [ "x${HAS_MPI}" == "xYES" ]; then
   ${MPI_EXEC} ${MPI_OPTS} ${NP} 2 ./${vftr_binary} || exit 1
else
   ./${vftr_binary} || exit 1
fi

if [ $has_gpu -eq 0 ]; then # On GPU

  echo "logfile_all: $logfile_all"
  echo "logfile_0: $logfile_0"
  echo "logfile_1: $logfile_1"
  bytes_h2d_all=`grep ${test_name}.c ${logfile_all} | head -1 | awk '{print $14}'`
  bytes_d2h_all=`grep ${test_name}.c ${logfile_all} | head -1 | awk '{print $16}'`
  bytes_ondevice_all=`grep ${test_name}.c ${logfile_all} | head -1 | awk '{print $18}'`

  bytes_h2d_0=`grep ${test_name}.c ${logfile_0} | head -1 | awk '{print $14}'`
  bytes_d2h_0=`grep ${test_name}.c ${logfile_0} | head -1 | awk '{print $16}'`
  bytes_ondevice_0=`grep ${test_name}.c ${logfile_0} | head -1 | awk '{print $18}'`

  bytes_h2d_1=`grep ${test_name}.c ${logfile_1} | head -1 | awk '{print $14}'`
  bytes_d2h_1=`grep ${test_name}.c ${logfile_1} | head -1 | awk '{print $16}'`
  bytes_ondevice_1=`grep ${test_name}.c ${logfile_1} | head -1 | awk '{print $18}'`

  bytes_h2d_sum=$((bytes_h2d_0 + bytes_h2d_1))
  bytes_d2h_sum=$((bytes_d2h_0 + bytes_d2h_1))
  bytes_ondevice_sum=$((bytes_ondevice_0 + bytes_ondevice_1))

  if [ "${bytes_h2d_all}" -ne "${bytes_h2d_sum}" ]; then
     echo "Bytes Host -> Device: Sum does not match"
     exit 1
  fi

  if [ "${bytes_d2h_all}" -ne "${bytes_d2h_sum}" ]; then
     echo "Bytes Device -> Host: Sum does not match"
     exit 1
  fi

  if [ "${bytes_ondevice_all}" -ne "${bytes_ondevice_sum}" ]; then
     echo "Bytes OnDevice: Sum does not match"
     exit 1
  fi

else # On Host
  grep "No OpenACC events have been registered" ${logfile}
  exit $?
fi
