#!/bin/bash
set -x
test_name=vftr_strip_module_names
output_file=${test_name}.out
ref_file=${srcdir}/ref_output/little_tasks.out
logfile=${test_name}_all.log
vfdfile=${test_name}_0.vfd

for file in ${output_file} ${logfile} ${vfdfile};
do
   if [ -f ${file} ] ; then
      rm ${file}
   fi
done

if [ "x${HAS_MPI}" == "xYES" ]; then
   ${MPI_EXEC} ${MPI_OPTS} ${NP} 1 ./${test_name} > ${output_file} || exit 1
else
   ./${test_name} > ${output_file} || exit 1
fi

diff ${output_file} ${ref_file} || exit 1

if [ ! -f ${logfile} ] ; then
   echo "Could not find logfile \"${logfile}\"!"
   exit 1
fi

module_name="little_ftasks"
nmodules=$(grep -i ${module_name} ${logfile} | wc -l)
if [ "${nmodules}" -ne "6" ] ; then
   echo "Expected to find the module \"${module_name}\" six times, "
   echo "but found it ${nmodules} times!"
   exit 1
fi

export VFTR_STRIP_MODULE_NAMES="yes"
for file in ${output_file} ${logfile} ${vfdfile};
do
   if [ -f ${file} ] ; then
      rm ${file}
   fi
done

if [ "x${HAS_MPI}" == "xYES" ]; then
   ${MPI_EXEC} ${MPI_OPTS} ${NP} 1 ./${test_name} > ${output_file} || exit 1
else
   ./${test_name} > ${output_file} || exit 1
fi

diff ${output_file} ${ref_file} || exit 1

if [ ! -f ${logfile} ] ; then
   echo "Could not find logfile \"${logfile}\"!"
   exit 1
fi

module_name="little_ftasks"
nmodules=$(grep -i ${module_name} ${logfile} | wc -l)
if [ "${nmodules}" -ne "0" ] ; then
   echo "Expected to find the module \"${module_name}\" zero times, "
   echo "but found it ${nmodules} times!"
   exit 1
fi
