#!/bin/bash
set -x

nvidia-smi
has_gpu=`echo $?`

test_name=callback_subscribe
exe_name=${test_name}
if [ $has_gpu -eq 0 ]; then
   output_file=${test_name}.out.gpu
else
   output_file=${test_name}.out.nogpu
fi
ref_file=${srcdir}/ref_output/${output_file}

rm -f ${output_file}

./${exe_name} > ${output_file} || exit 1
diff ${output_file} ${ref_file}
