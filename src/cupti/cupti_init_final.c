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

cudaError_t vftr_init_cupti (void (*cb_function) ()) {
  vftr_set_ngpus (); 
  if (vftrace.cupti_state.n_devices > 0) {
      cudaError_t ce;
      ce = cuptiSubscribe(&subscriber, (CUpti_CallbackFunc)cb_function, NULL);
      if (ce != cudaSuccess) return ce;
      ce = cuptiEnableDomain(1, subscriber, CUPTI_CB_DOMAIN_RUNTIME_API);
      return ce;
  }
  return cudaErrorNoDevice;
}

void vftr_finalize_cupti (collated_stacktree_t stacktree) {
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
