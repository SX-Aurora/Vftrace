#!/bin/bash

source ${srcdir}/../environment/filenames.sh

set -x

test_name=ve_counters_2
output_file=${test_name}.out
configfile=${test_name}.json
ref_file=${srcdir}/ref_output/${output_file}

cat << EOF > ${configfile}
{
  "mpi": {"show_table": false},
  "hwprof": {
     "counters": [
        {
           "hwc_name": "FPEC",
           "symbol": "f1"
        },
        {
           "hwc_name": "VX",
           "symbol": "f2"
        },
        {
           "hwc_name": "PCCC",
           "symbol": "f3"
        },
        {
           "hwc_name": "TTCC",
           "symbol": "f4"
        }
     ]
  } 
}
EOF
export VFTR_CONFIG=${configfile}

./${test_name} > ${output_file}

diff ${output_file} ${ref_file}
