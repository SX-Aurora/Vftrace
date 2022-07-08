#!/bin/bash
set -x
test_name=vftr_logfile_basename
output_file=vftr_logfile_basename.out
ref_file=${srcdir}/ref_output/little_tasks.out
logfile=${test_name}_0.log
vfdfile=${test_name}_0.vfd

for file in ${output_file} ${logfile} ${vfdfile};
do
   if [ -f ${file} ] ; then
      rm ${file}
   fi
done

if [ "x$HAS_MPI" == "xYES" ]; then
   ${MPI_EXEC} ${MPI_OPTS} ${NP} 1 ./${test_name} > ${output_file} || exit 1
else
   ./${test_name} > ${output_file} || exit 1
fi

diff ${output_file} ${ref_file} || exit 1

if [ ! -f ${logfile} ] ; then
   echo "Could not find logfile \"${logfile}\"!"
   exit 1
fi

testbasename="TestBasename"
export VFTR_LOGFILE_BASENAME=${testbasename}
logfile="${testbasename}_0.log"
vfdfile="${testbasename}_0.vfd"
for file in ${output_file} ${logfile} ${vfdfile};
do
   if [ -f ${file} ] ; then
      rm ${file}
   fi
done

if [ "x$HAS_MPI" == "xYES" ]; then
   ${MPI_EXEC} ${MPI_OPTS} ${NP} 1 ./${test_name} > ${output_file} || exit 1
else
   ./${test_name} > ${output_file} || exit 1
fi

diff ${output_file} ${ref_file} || exit 1

if [ ! -f ${logfile} ] ; then
   echo "Could not find logfile \"${logfile}\"!"
   exit 1
fi
