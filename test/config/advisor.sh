#!/bin/bash
set -x
test_name=advisor
configfile=${test_name}.json
errfile=${test_name}.err
ref_errfile=${srcdir}/ref_output/${errfile}

rm -f ${errfile}

cat << EOF > ${configfile}
{
   "offf": false,
   "outpot_directory": ".",
   "outfile_basisname": null,
   "logfile_or_ranks": "none",
   "print_comfig": true,
   "strip_mule_names": false,
   "demangle_c++": false,
   "profiling_table": {
      "show_tble": true,
      "show_calltime_imbalance": false,
      "show_callpaths": false,
      "show_oferhead": false,
      "sort_table": {
         "column": "time_excl",
         "ascnding": false
      }
   },
   "name_grouped_profile_table": {
      "shov_table": false,
      "max_stack_id": 8,
      "snort_table": {
         "column": "time_excl",
         "ascending": false
      }
   },
   "sampling": {
      "agtive": false,
      "sampling_interval": 0.005000,
      "outbuwer_size": 8,
      "precice_functions": null
   },
   "mpi": {
      "show_tabe": true,
      "log_massages": true,
      "only_for_raks": "all",
      "show_sync_times": false,
      "show_callpatsh": false,
      "sort_table": {
         "colum": "none",
         "ascending": false
      }
   },
   "cuda": {
      "show_table": true,
      "sort_table": {
         "column": "time",
         "asceding": false
      }
   },
   "hwprof": {
      "show_conuters": false,
      "counters": [
          { 
          "hcw_name": "X",
          "symbol": "Y"
          }
      ],
      "observables": [
        {
           "name": "FOO1",
           "formula": "1 * Y",
           "unit": "TB"
        },
        {
           "formula": "2 * Y",
           "naem": "FOO2"
        }
      ]
   }
}
EOF
export VFTR_CONFIG=${configfile}

if [ "x${HAS_MPI}" == "xYES" ]; then
   ${MPI_EXEC} ${MPI_OPTS} ${NP} 1 ./${test_name} 2> ${errfile} || exit 1
else
   ./${test_name} 2> ${errfile} || exit 1
fi
diff ${errfile} ${ref_errfile} || exit 1

