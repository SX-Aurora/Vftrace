#ifndef CUPTI_INIT_FINAL_H
#define CUPTI_INIT_FINAL_H

#include <cuda_runtime_api.h>

#include "collated_stack_types.h"

void vftr_set_ngpus();
cudaError_t vftr_init_cupti(void (*cb_function)());
void vftr_finalize_cupti(collated_stacktree_t stacktree);

#endif
