#!/bin/bash
set -x
test_name=vftr_show_calltime_imbalances
output_file=${test_name}.out
logfile=${test_name}_all.log
ref_file=${srcdir}/ref_output/little_tasks.out
imbalances_header=" Imbalances/% | on rank "
nranks=1

function rm_outfiles() {
   for file in ${output_file} ${test_name}_*.log ${test_name}_*.vfd;
   do
      if [ -f ${file} ] ; then
         rm ${file}
      fi
   done
}

function run_binary() {
   if [ "x${HAS_MPI}" == "xYES" ]; then
      ${MPI_EXEC} ${MPI_OPTS} ${NP} ${nranks} ./${test_name} > ${output_file} || exit 1
   else
      ./${test_name} > ${output_file} || exit 1
   fi
}

function check_logfile_exists() {
   logfile="${test_name}_$1.log"
   if [ ! -f ${logfile} ] ; then
      echo "Could not find logfile \"${logfile}\"!"
      exit 1
   fi
}

rm_outfiles
run_binary
diff ${output_file} ${ref_file} || exit 1
check_logfile_exists "all"
in_header=$(grep ${imbalances_header} ${logfile} | wc -l)
if [ "${in_header}" -ne "0" ] ; then
   echo "Found calltime imbalances in profile table although they should not be there!"
   exit 1
fi

export VFTR_SHOW_CALLTIME_IMBALANCES="on"
rm_outfiles
run_binary
diff ${output_file} ${ref_file} || exit 1
check_logfile_exists "all"
in_header=$(grep ${imbalances_header} ${logfile} | wc -l)
if [ "${in_header}" -ne "1" ] ; then
   echo "Could not find calltime imbalances in profile table although they should be there!"
   exit 1
fi

