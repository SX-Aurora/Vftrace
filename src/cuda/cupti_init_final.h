#ifndef CUDA_INIT_FINAL_H
#define CUDA_INIT_FINAL_H

#include <cuda_runtime_api.h>

#include "stack_types.h"

void vftr_set_ngpus();
cudaError_t vftr_init_cupti(void (*cb_function)());
void vftr_finalize_cupti(stacktree_t stacktree);

#endif
