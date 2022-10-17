#include <cuda_runtime_api.h>

#include "collated_stack_types.h"
#include "vftrace_state.h"

#include "cupti_vftr_callbacks.h"

static CUpti_SubscriberHandle subscriber; 

void vftr_set_ngpus () {
  int n_devices;
  cudaError_t ce = cudaGetDeviceCount(&n_devices);
  if (ce != cudaSuccess) {
     vftrace.cupti_state.n_devices = 0;
  } else { 
     vftrace.cupti_state.n_devices = n_devices;
  }
}

// When initializing CUPTI, we require the function pointer to the callback
// function as its argument. This allows to easily change between different callbacks
// e.g. for distinguishing between CUDA and OpenACC.
cudaError_t vftr_init_cupti (void (*cb_function) ()) {
  vftr_set_ngpus (); 
  if (vftrace.cupti_state.n_devices > 0) {
      cudaError_t ce;
      ce = cuptiSubscribe(&subscriber, (CUpti_CallbackFunc)cb_function, NULL);
      if (ce != cudaSuccess) return ce;
      // There are different domains supported by CUPTI:
      //   DRIVER, RUNTIME, RESOURCE, SYNCHRONIZE, NVTX
      // Currently, we only trace the user runtime calls.
      ce = cuptiEnableDomain(1, subscriber, CUPTI_CB_DOMAIN_RUNTIME_API);
      return ce;
  }
  return cudaErrorNoDevice;
}

void vftr_finalize_cupti (collated_stacktree_t stacktree) {
   // We go through the stacks and check if any CUDA events have
   // a non-success state.
   if (vftrace.cupti_state.n_devices > 0) {
      int n_warnings = 0;
      for (int istack = 0; istack < stacktree.nstacks; istack++) {
         collated_stack_t stack = stacktree.stacks[istack];
         cuptiprofile_t prof = stack.profile.cuptiprof;
         cudaError_t ce = cudaEventQuery (prof.start);
         if (ce != cudaSuccess) {
            if (n_warnings++ == 0) fprintf (stderr, "Warning: Some CUPTI events did not finish properly.\n");
            fprintf (stderr, "    CBID %d: %s (start)\n", prof.cbid, cudaGetErrorString(ce));
         }
         ce = cudaEventQuery (prof.stop);
         if (ce != cudaSuccess) {
            if (n_warnings++ == 0) fprintf (stderr, "Warning: Some CUPTI events did not finish properly.\n");
            fprintf (stderr, "    CBID %d: %s ()\n", prof.cbid, cudaGetErrorString(ce));
         }
      }


      cuptiUnsubscribe (subscriber);
   }
}
