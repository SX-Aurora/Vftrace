#ifndef CUPTI_INIT_FINAL_H
#define CUPTI_INIT_FINAL_H

#include <cuda_runtime_api.h>

void vftr_set_ngpus();
cudaError_t vftr_init_cupti();

#endif
