#ifndef COLLATED_CUDAPROFILING_H
#define COLLATED_CUDAPROFILING_H

#include <cupti.h>

#include "stack_types.h"
#include "collated_stack_types.h"

#include "collated_cudaprofiling_types.h"

collated_cudaprofile_t vftr_new_collated_cudaprofiling();
void vftr_collated_cudaprofiling_free (collated_cudaprofile_t *prof_ptr);

#endif
