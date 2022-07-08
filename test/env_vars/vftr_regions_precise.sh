#!/bin/bash
set -x
test_name=vftr_regions_precise
output_file=vftr_regions_precise.out
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

if [ -f ${logfile} ] ; then
   nreg=$(cat ${logfile} | grep "MyRegion<" | wc -l)
   if [ ${nreg} -ne "0" ] ; then
      echo "Expected \"MyRegion<\" to not appear in logfile. Found it ${nreg} times!"
      exit 1;
   fi
   nprecreg=$(cat ${logfile} | grep "MyRegion\*<" | wc -l)
   if [ ${nprecreg} -ne "1" ] ; then
      echo "Expected \"MyRegion*<\" to appear once in logfile. Found it ${nreg} times!"
      exit 1;
   fi
else
   echo "Could not find logfile \"${logfile}\"!"
   exit 1
fi

export VFTR_REGIONS_PRECISE="no"
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

if [ -f ${logfile} ] ; then
   nreg=$(cat ${logfile} | grep "MyRegion<" | wc -l)
   if [ ${nreg} -ne "1" ] ; then
      echo "Expected \"MyRegion\" to appear once in logfile. Found it ${nreg} times!"
      exit 1;
   fi
   nprecreg=$(cat ${logfile} | grep "MyRegion\*<" | wc -l)
   if [ ${nprecreg} -ne "0" ] ; then
      echo "Expected \"MyRegion*\" not to appear in logfile. Found it ${nreg} times!"
      exit 1;
   fi
else
   echo "Could not find logfile \"${logfile}\"!"
   exit 1
fi
