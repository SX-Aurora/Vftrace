#include <stdlib.h>
#include <stdio.h>

#include "vftrace_state.h"

#include "collated_cudaprofiling_types.h"
#include "cupti_vftr_callbacks.h"

collated_cudaprofile_t vftr_new_collated_cudaprofiling() {
  collated_cudaprofile_t prof;
  prof.cbid = 0;
  prof.n_calls[CUDA_COPY_IN] = 0;
  prof.n_calls[CUDA_COPY_OUT] = 0;
  prof.t_ms = 0;
  prof.memcpy_bytes[CUDA_COPY_IN] = 0;
  prof.memcpy_bytes[CUDA_COPY_OUT] = 0; 
  prof.overhead_nsec = 0;
  prof.on_nranks = 0;
  memset (prof.avg_ncalls, 0, 2 * sizeof(int));
  memset (prof.max_ncalls, 0, 2 * sizeof(int));
  memset (prof.max_on_rank, 0, 2 * sizeof(int));
  memset (prof.min_ncalls, 0, 2 * sizeof(int));
  memset (prof.min_on_rank, 0, 2 * sizeof(int));
  return prof;
}

void vftr_collated_cudaprofiling_free(collated_cudaprofile_t *prof_ptr) {
}
