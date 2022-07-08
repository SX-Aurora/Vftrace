#!/bin/bash
set -x
test_name=vftr_off
output_file=vftr_off.out
ref_file=${srcdir}/ref_output/little_tasks.out
logfile=${test_name}_0.log
vfdfile=${test_name}_0.vfd

for file in ${output_file} ${logfile} ${vfdfile};
do
   if [ -f ${file} ] ; then
      rm ${file}
   fi
done

export VFTR_OFF="yes"

if [ "x$HAS_MPI" == "xYES" ]; then
   ${MPI_EXEC} ${MPI_OPTS} ${NP} 1 ./${test_name} > ${output_file} || exit 1
else
   ./${test_name} > ${output_file} || exit 1
fi

diff ${output_file} ${ref_file} || exit 1

if [ -f ${logfile} ] ; then
   echo "Logfile \"${logfile}\" was created although vftrace shold be turned off!"
   exit 1
fi
if [ -f ${vfdfile} ] ; then
   echo "VFD-File \"${vfdfile}\" was created although vftrace shold be turned off!"
   exit 1
fi
