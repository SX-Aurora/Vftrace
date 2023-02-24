#!/bin/bash
set -x

nvidia-smi
has_gpu=`echo $?`

source ${srcdir}/../environment/filenames.sh

test_name=acc_region5
vftr_binary=${test_name}
configfile=${test_name}.json

determine_bin_prefix $vftr_binary

logfile=$(get_logfile_name ${vftr_binary} "all")

cat << EOF > ${configfile}
{
   "openacc": {
        "show_event_details": false
    }
}
EOF
export VFTR_CONFIG=${configfile}

if [ "x${HAS_MPI}" == "xYES" ]; then
   ${MPI_EXEC} ${MPI_OPTS} ${NP} 1 ./${vftr_binary} || exit 1
else
   ./${vftr_binary} || exit 1
fi

if [ $has_gpu -eq 0 ]; then # On GPU

  bytes_h2d=`grep ${test_name}.c ${logfile} | head -1 | awk '{print $14}'`
  bytes_d2h=`grep ${test_name}.c ${logfile} | head -1 | awk '{print $16}'`
  
  if [ "${bytes_h2d}" -ne 40 ]; then
     echo "Bytes Host -> Device does not match."
     exit 1;
  fi
  
  if [ "${bytes_d2h}" -ne 0 ]; then
     echo "Bytes Device -> Host does not match."
     exit 1;
  fi
  
  bytes_h2d=`grep ${test_name}.c ${logfile} | tail -1 | awk '{print $14}'`
  bytes_d2h=`grep ${test_name}.c ${logfile} | tail -1 | awk '{print $16}'`
  
  if [ "${bytes_h2d}" -ne 0 ]; then
     echo "Bytes Host -> Device does not match."
     exit 1;
  fi
  
  if [ "${bytes_d2h}" -ne 40 ]; then
     echo "Bytes Device -> Host does not match."
     exit 1;
  fi
  
else # On Host
  grep "No OpenACC events have been registered" ${logfile}
  exit $?
fi
