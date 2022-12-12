#!/bin/bash

source ${srcdir}/../environment/filenames.sh

set -x
test_name=strip_module_names
configfile=${test_name}.json
output_file=${test_name}.out
ref_file=${srcdir}/ref_output/little_tasks.out
determine_bin_prefix $test_name
logfile=$(get_logfile_name $test_name "all")

export VFTR_CONFIG=${configfile}
echo "{\"papi\": {\"show_tables\": false}}" > ${configfile}

rm_outfiles $output_file "" $test_name
if [ "x${HAS_MPI}" == "xYES" ]; then
   ${MPI_EXEC} ${MPI_OPTS} ${NP} 1 ./${test_name} > ${output_file} || exit 1
else
   ./${test_name} > ${output_file} || exit 1
fi
diff ${output_file} ${ref_file} || exit 1
check_file_exists $logfile
module_name="little_ftasks"
nmodules=$(grep -i ${module_name} ${logfile} | wc -l)
if [ "${nmodules}" -ne "6" ] ; then
   echo "Expected to find the module \"${module_name}\" six times, "
   echo "but found it ${nmodules} times!"
   exit 1
fi

echo "{\"strip_module_names\": true, \"papi\": {\"show_tables\": false}}" > ${configfile}
rm_outfiles $output_file "" $test_name
if [ "x${HAS_MPI}" == "xYES" ]; then
   ${MPI_EXEC} ${MPI_OPTS} ${NP} 1 ./${test_name} > ${output_file} || exit 1
else
   ./${test_name} > ${output_file} || exit 1
fi
diff ${output_file} ${ref_file} || exit 1
check_file_exists $logfile
module_name="little_ftasks"
nmodules=$(grep -i ${module_name} ${logfile} | wc -l)
if [ "${nmodules}" -ne "0" ] ; then
   echo "Expected to find the module \"${module_name}\" zero times, "
   echo "but found it ${nmodules} times!"
   exit 1
fi
