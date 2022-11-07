#ifndef CUDA_UTILS_H
#define CUDA_UTILS_H

#include <stdio.h>
#include <stdbool.h>

void vftr_show_used_gpu_info (FILE *fp);

bool vftr_cuda_cbid_belongs_to_class (int cbid, int cbid_class);

#endif
