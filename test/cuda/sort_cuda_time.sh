#!/bin/bash
set -x
test_name=sort_cuda_time
exe_name=print_cuda_table
output_file=${test_name}.out
ref_file=${srcdir}/ref_output/${output_file}

rm -f ${output_file}

nvidia-smi
has_gpu=`echo $?`

export VFTR_SORT_CUDA_TABLE="TIME"
./${exe_name} > ${output_file} || exit 1

if [ $has_gpu -eq 0 ]; then
### nvidia-smi has identified GPUs. Remove the corresponding line
  sed -i '/Using [0-9+] GPUs/d' ${output_file}
  sed -i '/Visible GPUs/d' ${output_file}
else
### No GPUs found
   sed -i '/No GPUs available/d' ${output_file}
fi

diff ${output_file} ${ref_file}
