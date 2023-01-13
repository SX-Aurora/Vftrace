#!/bin/bash

source ${srcdir}/../environment/filenames.sh

set -x
test_name=print_config_default_scenario
output_file=${test_name}.out
configfile=${test_name}.json
ref_file=${srcdir}/ref_output/${output_file}
default_scenario=$PWD/scenario.json

cat << EOF > ${default_scenario}
{
      "counters": [
          {
          "hwc_name": "READ_BYTES",
          "symbol": "B"
          },
          {
          "hwc_name": "WRITE_BYTES",
          "symbol": "W"
          }
      ],
      "observables": [
         {
            "name": "Read I/O",
            "formula": "B / T / 1024 / 1024 / 1024",
            "unit": "GB / s"
         }, 
         {
            "name": "Write I/O",
            "formula": "W / T / 1024 / 1024 / 1024",
            "unit": "GB / s"
         }
      ]
}
EOF

cat << EOF > ${configfile}
{
   "print_config": true,
   "papi": {
       "default_scenario": "$default_scenario"
   }
}
EOF
export VFTR_CONFIG=${configfile}

if [ "x${HAS_MPI}" == "xYES" ]; then
   ${MPI_EXEC} ${MPI_OPTS} ${NP} 1 ./${test_name} > ${output_file} || exit 1
else
   ./${test_name} > ${output_file} || exit 1
fi

# default_scenario includes the pwd string, it needs to be removed. 
sed -i '/default_scenario/d' ${output_file}
diff ${output_file} ${ref_file} || exit 1
