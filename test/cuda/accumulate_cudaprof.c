#include <stdlib.h>
#include <stdio.h>

#include "stack_types.h"
#include "collated_stack_types.h"
#include "collate_stacks.h"
#include "collate_profiles.h"
#include "callprofiling_types.h"

#include "cupti_vftr_callbacks.h"
#include "cudaprofiling_types.h"

#include "dummy_stacktree.h"

// COMPUTE_CBIDS: CUPTI_RUNTIME_TRACE_CBID_cudaLaunch_v3020,
//                CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000
// MEMCPY_CBIDS:  CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy_v3020,
//                CUPTI_RUNTIME_TRACE_CBID_cudaMemcpyAsync_v3020

int main(int argc, char **argv) {

   vftr_init_dummy_stacktree (10);
   vftr_register_dummy_call_stack ("func0<init", 1);
   vftr_register_dummy_call_stack ("cudafunc1<init", 2);
   vftr_register_dummy_cuda_stack ("cudafunc1<init", CUPTI_RUNTIME_TRACE_CBID_cudaLaunch_v3020,
                                    1000.0, CUDA_NOCOPY, 0);
   vftr_register_dummy_cuda_stack ("cudafunc2<func0<init", CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy_v3020,
				    1000.0, CUDA_COPY_IN, 1048576);
   vftr_register_dummy_cuda_stack ("cudafunc3<func0<init", CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy_v3020,
				    20000.0, CUDA_COPY_OUT, 524288);
   vftr_register_dummy_call_stack ("cudafunc1<init", 2);
   for (int i = 0; i < 4; i++) {
      vftr_register_dummy_cuda_stack ("cudafunc1<init", CUPTI_RUNTIME_TRACE_CBID_cudaLaunch_v3020,
                                       1000.0, CUDA_NOCOPY, 0);
   }
   for (int i = 0; i < 3; i++) {
      vftr_register_dummy_cuda_stack ("cudafunc4<func0<init", CUPTI_RUNTIME_TRACE_CBID_cudaLaunch_v3020,
				       10000.0, CUDA_NOCOPY, 0);
   }



   stacktree_t stacktree = vftr_get_dummy_stacktree();
   
   for (int istack = 0; istack < stacktree.nstacks; istack++) {
      vftr_stack_t this_stack = stacktree.stacks[istack];
      profile_t *this_profile = this_stack.profiling.profiles;
      callprofile_t callprof = this_profile->callprof; 
      cudaprofile_t cudaprof = this_profile->cudaprof;
      printf ("istack: %d\n", istack);
      printf ("localID: %d\n", this_stack.lid);
      printf ("name: %s\n", this_stack.name);
      printf ("callprof.t_excl: %lld\n", callprof.time_excl_nsec);
      printf ("callprof.calls: %d\n", callprof.calls);
      printf ("cudaprof.cbid: %d\n", cudaprof.cbid);
      printf ("cudaprof.calls: %d\n", cudaprof.n_calls[0] + cudaprof.n_calls[1]);
      printf ("cudaprof.t_ms: %.3f\n", cudaprof.t_ms);
      printf ("cudaprof.memcpy_in: %lld\n", cudaprof.memcpy_bytes[CUDA_COPY_IN]);
      printf ("cudaprof.memcpy_out: %lld\n", cudaprof.memcpy_bytes[CUDA_COPY_OUT]);
      printf ("------------------------------------\n");
   }
}
