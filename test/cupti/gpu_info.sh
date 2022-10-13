#!/bin/bash

set -x

nvidia-smi
has_gpu=`echo $?`

check_cuda=`./gpu_info` || exit 1
### When no GPU has been found by nvidia-smi, the return value is anything but 0. gpu_info must return 0.
if [ $has_gpu -ne 0 ] && [ $check_cuda -eq 0 ]; then
   exit 0
### If nvidia-smi has identified GPUs, the return values is 0. gpu_info must return 1.
elif [ $has_gpu -eq 0 ] && [ $check_cuda -eq 1 ]; then
   exit 0
### No fitting situation (nvidia-smi finds GPUs, but our implementation fails to do so).
else
   exit 1
fi
