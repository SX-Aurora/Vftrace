#include <cuda_runtime_api.h>

#include "collated_stack_types.h"
#include "vftrace_state.h"

#include "cupti_vftr_callbacks.h"

static CUpti_SubscriberHandle subscriber; 

void vftr_set_ngpus () {
  int n_devices;
  cudaError_t ce = cudaGetDeviceCount(&n_devices);
  if (ce != cudaSuccess) {
     vftrace.cuda_state.n_devices = 0;
  } else { 
     vftrace.cuda_state.n_devices = n_devices;
  }
}

// When initializing CUpti, we require the function pointer to the callback
// function as its argument. This allows to easily change between different callbacks
// e.g. for distinguishing between CUDA and OpenACC.
cudaError_t vftr_init_cupti (void (*cb_function) ()) {
  vftr_set_ngpus (); 
  if (vftrace.cuda_state.n_devices > 0) {
      cudaError_t ce;
      ce = cuptiSubscribe(&subscriber, (CUpti_CallbackFunc)cb_function, NULL);
      if (ce != cudaSuccess) return ce;
      // There are different domains supported by CUpti:
      //   DRIVER, RUNTIME, RESOURCE, SYNCHRONIZE, NVTX
      // Currently, we only trace the user runtime calls.
      ce = cuptiEnableDomain(1, subscriber, CUPTI_CB_DOMAIN_RUNTIME_API);
      return ce;
  }
  return cudaErrorNoDevice;
}

void vftr_finalize_cupti (stacktree_t stacktree) {
   // We go through the stacks and check if any CUDA events have
   // a non-success state.
   if (vftrace.cuda_state.n_devices > 0) {
      int n_warnings = 0;
      for (int istack = 0; istack < stacktree.nstacks; istack++) {
         vftr_stack_t stack = stacktree.stacks[istack];
         cudaprofile_t prof = stack.profiling.profiles[0].cudaprof;
         //printf ("istack: %d\n", istack);
         //fflush(stdout);
         //cudaError_t ce = cudaEventQuery (prof.start);
         //if (ce != cudaSuccess) {
         //   if (n_warnings++ == 0) fprintf (stderr, "Warning: Some CUpti events did not finish properly.\n");
         //   fprintf (stderr, "    stack: %d, rank: %d, CBID %d: %s (start)\n", istack, vftrace.process.processID, prof.cbid, cudaGetErrorString(ce));
         //}
         //ce = cudaEventQuery (prof.stop);
         //if (ce != cudaSuccess) {
         //   if (n_warnings++ == 0) fprintf (stderr, "Warning: Some CUpti events did not finish properly.\n");
         //   fprintf (stderr, "    stack: %d, rank: %d, CBID %d: %s (end)\n", istack, vftrace.process.processID, prof.cbid, cudaGetErrorString(ce));
         //}
      }


      cuptiUnsubscribe (subscriber);
   }
}
