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
  return prof;
}

void vftr_accumulate_cuptiprofiling (cuptiprofile_t *prof, int cbid, int n_calls,
                                     float t_ms, int mem_dir, uint64_t memcpy_bytes) {
   prof->cbid = cbid;
   prof->n_calls += n_calls;
   prof->t_ms += t_ms;
   if (mem_dir != CUPTI_NOCOPY) prof->memcpy_bytes[mem_dir] += memcpy_bytes;
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
