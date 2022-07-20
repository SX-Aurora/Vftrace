#!/bin/bash
set -x
test_name=vftr_out_directory
output_file=vftr_out_directory.out
ref_file=${srcdir}/ref_output/little_tasks.out
logfile=${test_name}_0.log
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

outdir="$(pwd)/testoutdir/"
mkdir -p ${outdir}
export VFTR_OUT_DIRECTORY=${outdir}
logfile="${outdir}/${test_name}_0.log"
vfdfile="${outdir}/${test_name}_0.vfd"
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
if [ -d ${outdir} ] ; then
   rm -rf ${outdir}
fi
