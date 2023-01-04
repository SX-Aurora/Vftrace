#!/bin/bash
set -x
test_name=collate_cudaprofiles_parallel
exe_name=collate_cudaprofiles_parallel
output_file=${test_name}.out
ref_file=${srcdir}/ref_output/${output_file}

rm -f ${output_file}

nvidia-smi
has_gpu=`echo $?`

${MPI_EXEC} ${MPI_OPTS} ${NP} 2 ./${exe_name} > ${output_file} || exit 1

if [ $has_gpu -eq 0 ]; then
### nvidia-smi has identified GPUs. Remove the corresponding line
  sed -i '/Using [0-9+] GPUs/d' ${output_file}
  sed -i '/Visible GPUs/d' ${output_file}
else
### No GPUs found
   sed -i '/No GPUs available/d' ${output_file}
fi

diff ${output_file} ${ref_file}
