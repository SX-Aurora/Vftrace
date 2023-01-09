#!/bin/bash

source ${srcdir}/../environment/filenames.sh

set -x
test_name=print_config
output_file=${test_name}.out
configfile=${test_name}.json
logfile=${test_name}_all.log
ref_file=${srcdir}/ref_output/${logfile}.trunc

cat << EOF > ${configfile}
{
   "print_config": true,
   "papi": {
       "show_counters": true,
       "sort_by_column": 0,
       "counters": [
           {
           "hwc_name": "perf::CYCLES",
           "symbol": "f1"
           },
           {
           "hwc_name": "FP_ARITH:SCALAR_SINGLE",
           "symbol": "fpsingle"
           },
           {
           "hwc_name": "FP_ARITH:SCALAR_DOUBLE",
           "symbol": "fpdouble"
           }
       ],
       "observables": [
           {
              "name": "f",
              "formula": "f1 / T * 1e-6",
              "unit": "MHz"
           },
           {
              "name": "perf",
              "formula": "(fpsingle + fpdouble) / T * 1e-6",
              "unit": "MFlop/s"
           }
       ]
    } 
}
EOF
export VFTR_CONFIG=${configfile}

if [ "x${HAS_MPI}" == "xYES" ]; then
   ${MPI_EXEC} ${MPI_OPTS} ${NP} 1 ./${test_name} > ${output_file} || exit 1
else
   ./${test_name} > ${output_file} || exit 1
fi

sed -n '/Vftrace configuration read from/,$p' ${logfile} > ${logfile}.trunc

diff ${logfile}.trunc ${ref_file} || exit 1
