#!/bin/bash
set -x
test_name=vftr_logfile_for_ranks_assert
output_file=${test_name}.out
error_file=${test_name}.err
ref_file=${srcdir}/ref_output/little_tasks.out
nranks=1

function run_binary() {
   if [ "x${HAS_MPI}" == "xYES" ]; then
      ${MPI_EXEC} ${MPI_OPTS} ${NP} ${nranks} ./${test_name} \
         > ${output_file} 2> ${error_file} || exit 1
   else
      ./${test_name} > ${output_file} 2> ${error_file} || exit 1
   fi
}

function rm_outfiles() {
   for file in ${output_file} ${error_file} ${test_name}_*.log ${test_name}_*.vfd;
   do
      if [ -f ${file} ] ; then
         rm ${file}
      fi
   done
}

rm_outfiles
run_binary
diff ${output_file} ${ref_file} || exit 1
cat ${error_file}
nwarn=$(grep "Warning" ${error_file} | wc -l)
if [ "${nwarn}" -ne 0 ] ; then
   echo "Expected no warning for environment assertion"
   exit 1
fi

export VFTR_LOGFILE_FOR_RANKS="3-5,7,8,15-120,40-65"
rm_outfiles
run_binary
cat ${error_file}
nwarn=$(grep "Warning" ${error_file} | wc -l)
if [ "${nwarn}" -ne 0 ] ; then
   echo "Expected no warning for environment assertion"
   exit 1
fi

export VFTR_LOGFILE_FOR_RANKS="all"
rm_outfiles
run_binary
cat ${error_file}
nwarn=$(grep "Warning" ${error_file} | wc -l)
if [ "${nwarn}" -ne 0 ] ; then
   echo "Expected no warning for environment assertion"
   exit 1
fi

export VFTR_LOGFILE_FOR_RANKS="none"
rm_outfiles
run_binary
cat ${error_file}
nwarn=$(grep "Warning" ${error_file} | wc -l)
if [ "${nwarn}" -ne 0 ] ; then
   echo "Expected no warning for environment assertion"
   exit 1
fi

export VFTR_LOGFILE_FOR_RANKS="nonsense"
rm_outfiles
run_binary
cat ${error_file}
nwarn=$(grep "Warning" ${error_file} | wc -l)
if [ "${nwarn}" -ne 1 ] ; then
   echo "Expected warning for environment assertion"
   exit 1
fi

export VFTR_LOGFILE_FOR_RANKS="1-2,3,-5-6,17"
rm_outfiles
run_binary
cat ${error_file}
nwarn=$(grep "Warning" ${error_file} | wc -l)
if [ "${nwarn}" -ne 1 ] ; then
   echo "Expected warning for environment assertion"
   exit 1
fi
