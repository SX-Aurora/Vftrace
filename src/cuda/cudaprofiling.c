#include <stdlib.h>
#include <stdio.h>

#include "vftrace_state.h"

#include "cudaprofiling_types.h"
#include "cupti_vftr_callbacks.h"

cudaprofile_t vftr_new_cudaprofiling() {
  cudaprofile_t prof;
  if (vftrace.cuda_state.n_devices > 0) {
// When called on a system without GPUs, this will silently return
// cudaErrorNoDevice.
     cudaEventCreate (&(prof.start));
     cudaEventCreate (&(prof.stop));
  }
  prof.cbid = 0;
  prof.n_calls = 0;
  prof.t_ms = 0;
  prof.memcpy_bytes[CUDA_COPY_IN] = 0;
  prof.memcpy_bytes[CUDA_COPY_OUT] = 0; 
  prof.overhead_nsec = 0;
  return prof;
}

void vftr_accumulate_cudaprofiling (cudaprofile_t *prof, int cbid, int n_calls,
                                    float t_ms, int mem_dir, uint64_t memcpy_bytes) {
   prof->cbid = cbid;
   prof->n_calls += n_calls;
   prof->t_ms += t_ms;
   if (mem_dir != CUDA_NOCOPY) prof->memcpy_bytes[mem_dir] += memcpy_bytes;
}

void vftr_accumulate_cudaprofiling_overhead (cudaprofile_t *prof, long long t_nsec) {
   prof->overhead_nsec += t_nsec;
}

// The other overhead routines loop over the number of threads.
// Since CUDA profiling is only supported for one thread, we don't need that (yet).
long long vftr_get_total_cuda_overhead (stacktree_t stacktree) {
   long long overhead_nsec = 0;
   
   for (int istack = 0; istack < stacktree.nstacks; istack++) {
	stack_t *stack = stacktree.stacks + istack;
        profile_t *prof = stack->profiling.profiles;
        cudaprofile_t *cudaprof = &(prof->cudaprof);
        overhead_nsec += cudaprof->overhead_nsec;
   }
   return overhead_nsec;
}

long long vftr_get_total_collated_cuda_overhead (collated_stacktree_t stacktree) {
   long long overhead_nsec = 0;
   
   for (int istack = 0; istack < stacktree.nstacks; istack++) {
	collated_stack_t *stack = stacktree.stacks + istack;
        cudaprofile_t *cudaprof = &(stack->profile.cudaprof);
        overhead_nsec += cudaprof->overhead_nsec;
   }
   return overhead_nsec;
} 

cudaprofile_t vftr_add_cudaprofiles(cudaprofile_t profA, cudaprofile_t profB) {
   cudaprofile_t profC;
   profC.cbid = profA.cbid; // The CBIDs of both profiles are identical.
   profC.n_calls = profA.n_calls + profB.n_calls;
   profC.t_ms = profA.t_ms + profB.t_ms;
   profC.memcpy_bytes[CUDA_COPY_IN] = profA.memcpy_bytes[CUDA_COPY_IN] + profB.memcpy_bytes[CUDA_COPY_IN];
   profC.memcpy_bytes[CUDA_COPY_OUT] = profA.memcpy_bytes[CUDA_COPY_OUT] + profB.memcpy_bytes[CUDA_COPY_OUT];
   return profC;
}

void vftr_cudaprofiling_free(cudaprofile_t *prof_ptr) {
  if (vftrace.cuda_state.n_devices > 0 && prof_ptr->cbid > 0) {
     cudaEventDestroy (prof_ptr->start);
     cudaEventDestroy (prof_ptr->stop);
  }
}
