#!/bin/bash

source ${srcdir}/../environment/filenames.sh

set -x
test_name=vftr_show_calltime_imbalances
output_file=${test_name}.out
ref_file=${srcdir}/ref_output/little_tasks.out
imbalances_header=" Imbalances/% | on rank "
nranks=1

logfile=$(get_logfile_name $test_name "all")

function run_binary() {
   if [ "x${HAS_MPI}" == "xYES" ]; then
      ${MPI_EXEC} ${MPI_OPTS} ${NP} ${nranks} ./${test_name} > ${output_file} || exit 1
   else
      ./${test_name} > ${output_file} || exit 1
   fi
}

rm_outfiles $output_file "" $test_name
run_binary
diff ${output_file} ${ref_file} || exit 1
check_file_exists $logfile
in_header=$(grep ${imbalances_header} ${logfile} | wc -l)
if [ "${in_header}" -ne "0" ] ; then
   echo "Found calltime imbalances in profile table although they should not be there!"
   exit 1
fi

export VFTR_SHOW_CALLTIME_IMBALANCES="on"
rm_outfiles $output_file "" $test_name
run_binary
diff ${output_file} ${ref_file} || exit 1
check_file_exists $logfile
in_header=$(grep ${imbalances_header} ${logfile} | wc -l)
if [ "${in_header}" -ne "1" ] ; then
   echo "Could not find calltime imbalances in profile table although they should be there!"
   exit 1
fi

