#ifndef CUDAPROFILING_H
#define CUDAPROFILING_H

#include <cupti.h>

#include "stack_types.h"
#include "collated_stack_types.h"

#include "cudaprofiling_types.h"

cudaprofile_t vftr_new_cudaprofiling();
void vftr_cudaprofiling_free (cudaprofile_t *prof_ptr);

void vftr_accumulate_cudaprofiling (cudaprofile_t *prof, int cbid, int n_calls,
                                    float t_ms, int mem_dir, uint64_t memcpy_bytes);

void vftr_accumulate_cudaprofiling_overhead (cudaprofile_t *prof, long long t_nsec);

long long vftr_get_total_cuda_overhead (stacktree_t stacktree);
long long vftr_get_total_collated_cuda_overhead (collated_stacktree_t stacktree);

cudaprofile_t vftr_add_cudaprofiles(cudaprofile_t profA, cudaprofile_t profB);

#endif
