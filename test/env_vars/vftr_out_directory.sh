#!/bin/bash

source ${srcdir}/../environment/filenames.sh

set -x
test_name=vftr_out_directory
output_file=vftr_out_directory.out
ref_file=${srcdir}/ref_output/little_tasks.out
logfile=$(get_logfile_name $test_name "all")

rm_outfiles $output_file "" $test_name

if [ "x${HAS_MPI}" == "xYES" ]; then
   ${MPI_EXEC} ${MPI_OPTS} ${NP} 1 ./${test_name} > ${output_file} || exit 1
else
   ./${test_name} > ${output_file} || exit 1
fi

diff ${output_file} ${ref_file} || exit 1

check_file_exists $logfile

outdir="$(pwd)/testoutdir/"
mkdir -p ${outdir}
export VFTR_OUT_DIRECTORY=${outdir}
rm_outfiles $output_file "" $test_name

if [ "x${HAS_MPI}" == "xYES" ]; then
   ${MPI_EXEC} ${MPI_OPTS} ${NP} 1 ./${test_name} > ${output_file} || exit 1
else
   ./${test_name} > ${output_file} || exit 1
fi

diff ${output_file} ${ref_file} || exit 1

if [ -d ${outdir} ]; then
   cd $outdir
   check_file_exists $logfile
   cd ..
   rm -r ${outdir}
else
   echo "Could not find output directory \"${outdir}\"!" 
fi
