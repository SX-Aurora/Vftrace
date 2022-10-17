#!/bin/bash
set -x
test_name=sort_cupti_memcpy
exe_name=print_cupti_table
output_file=${test_name}.out
ref_file=${srcdir}/ref_output/${output_file}

rm -f ${output_file}

nvidia-smi
has_gpu=`echo $?`

export VFTR_SORT_CUPTI_TABLE="MEMCPY"
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
