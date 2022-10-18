#include <stdlib.h>
#include <stdio.h>

#include "vftrace_state.h"

#include "cuptiprofiling_types.h"
#include "cupti_vftr_callbacks.h"

cuptiprofile_t vftr_new_cuptiprofiling() {
  cuptiprofile_t prof;
  if (vftrace.cupti_state.n_devices > 0) {
// When called on a system without GPUs, this will silently return
// cudaErrorNoDevice.
     cudaEventCreate (&(prof.start));
     cudaEventCreate (&(prof.stop));
  }
  prof.cbid = 0;
  prof.n_calls = 0;
  prof.t_ms = 0;
  prof.memcpy_bytes[CUPTI_COPY_IN] = 0;
  prof.memcpy_bytes[CUPTI_COPY_OUT] = 0; 
  prof.overhead_nsec = 0;
  return prof;
}

void vftr_accumulate_cuptiprofiling (cuptiprofile_t *prof, int cbid, int n_calls,
                                     float t_ms, int mem_dir, uint64_t memcpy_bytes) {
   prof->cbid = cbid;
   prof->n_calls += n_calls;
   prof->t_ms += t_ms;
   if (mem_dir != CUPTI_NOCOPY) prof->memcpy_bytes[mem_dir] += memcpy_bytes;
}

void vftr_accumulate_cuptiprofiling_overhead (cuptiprofile_t *prof, long long t_nsec) {
   prof->overhead_nsec += t_nsec;
}

// The other overhead routines loop over the number of threads.
// Since CUPTI is only supported for one thread, we don't need that (yet).
long long vftr_get_total_cupti_overhead (stacktree_t stacktree) {
   long long overhead_nsec = 0;
   
   for (int istack = 0; istack < stacktree.nstacks; istack++) {
	stack_t *stack = stacktree.stacks + istack;
        profile_t *prof = stack->profiling.profiles;
        cuptiprofile_t *cuptiprof = &(prof->cuptiprof);
        overhead_nsec += cuptiprof->overhead_nsec;
   }
   return overhead_nsec;
}

long long vftr_get_total_collated_cupti_overhead (collated_stacktree_t stacktree) {
   long long overhead_nsec = 0;
   
   for (int istack = 0; istack < stacktree.nstacks; istack++) {
	collated_stack_t *stack = stacktree.stacks + istack;
        cuptiprofile_t *cuptiprof = &(stack->profile.cuptiprof);
        overhead_nsec += cuptiprof->overhead_nsec;
   }
   return overhead_nsec;
} 

cuptiprofile_t vftr_add_cuptiprofiles(cuptiprofile_t profA, cuptiprofile_t profB) {
   cuptiprofile_t profC;
   profC.cbid = profA.cbid + profB.cbid;
   profC.n_calls = profA.n_calls + profB.n_calls;
   profC.t_ms = profA.t_ms + profB.t_ms;
   profC.memcpy_bytes[CUPTI_COPY_IN] = profA.memcpy_bytes[CUPTI_COPY_IN] + profB.memcpy_bytes[CUPTI_COPY_IN];
   profC.memcpy_bytes[CUPTI_COPY_OUT] = profA.memcpy_bytes[CUPTI_COPY_OUT] + profB.memcpy_bytes[CUPTI_COPY_OUT];
   return profC;
}

void vftr_cuptiprofiling_free(cuptiprofile_t *prof_ptr) {
  if (vftrace.cupti_state.n_devices > 0 && prof_ptr->cbid > 0) {
     cudaEventDestroy (prof_ptr->start);
     cudaEventDestroy (prof_ptr->stop);
  }
}
