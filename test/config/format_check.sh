#!/bin/bash
set -x
test_name=format_check
configfile=${test_name}.json
errfile=${test_name}.err
ref_errfile=${srcdir}/ref_output/${errfile}

rm -f ${errfile}

cat << EOF > ${configfile}
{
   "off": false,
   "output_directory"
   "outfile_basename": null,
   "logfile_for_ranks": "none",
   "print_config": true,
   "strip_module_names": false,
   "demangle_cxx": false,
   "profile_table": {
      "show_table": true,
      "show_calltime_imbalances": false,
      "show_callpath": false,
      "show_overhead": false,
      "sort_table": {
         "column": "time_excl",
         "ascending": false
      }
   }
}
EOF
export VFTR_CONFIG=${configfile}

echo "THIS CODE ABORTION IS INTENDED!"
if [ "x${HAS_MPI}" == "xYES" ]; then
   ${MPI_EXEC} ${MPI_OPTS} ${NP} 1 ./${test_name} 2> ${errfile} && exit 1
else
   ./${test_name} 2> ${errfile} && exit 1
fi
echo "" >> ${errfile}

cat << EOF > ${configfile}
{
   "off": false,
   "output_directory": ".",
   "outfile_basename": null,
   "logfile_for_ranks": "none",
   "print_config": true,
   "strip_module_names": false,
   "demangle_cxx": false,,
   "profile_table": {
      "show_table": true,
      "show_calltime_imbalances": false,
      "show_callpath": false,
      "show_overhead": false,
      "sort_table": {
         "column": "time_excl",
         "ascending": false
      }
   }
}
EOF

echo "THIS CODE ABORTION IS INTENDED!"
if [ "x${HAS_MPI}" == "xYES" ]; then
   ${MPI_EXEC} ${MPI_OPTS} ${NP} 1 ./${test_name} 2>> ${errfile} && exit 1
else
   ./${test_name} 2>> ${errfile} && exit 1
fi
echo "" >> ${errfile}

cat << EOF > ${configfile}
{
   "off": false,
   "output_directory": ".",
   "outfile_basename": null,
   "logfile_for_ranks": "none",
   "print_config": true,
   "strip_module_names": false,
   "demangle_cxx": false,
   "profile_table": {
      "show_table": true,
      "show_calltime_imbalances": false,
      "show_callpath": false,
      "show_overhead": false,
      "sort_table": {
         "column": "time_excl",
         "ascending": false
      },
   }
}
EOF

echo "THIS CODE ABORTION IS INTENDED!"
if [ "x${HAS_MPI}" == "xYES" ]; then
   ${MPI_EXEC} ${MPI_OPTS} ${NP} 1 ./${test_name} 2>> ${errfile} && exit 1
else
   ./${test_name} 2>> ${errfile} && exit 1
fi

diff ${errfile} ${ref_errfile} || exit 1
