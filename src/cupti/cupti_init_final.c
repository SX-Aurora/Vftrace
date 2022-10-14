#include <cuda_runtime_api.h>

#include "vftrace_state.h"
#include "callbacks.h"

void vftr_set_ngpus () {
  int n_devices;
  cudaError_t ce = cudaGetDeviceCount(&n_devices);
  if (ce != cudaSuccess) {
     vftrace.cupti_state.n_devices = 0;
  } else { 
     vftrace.cupti_state.n_devices = n_devices;
  }
}

cudaError_t vftr_init_cupti () {
  vftr_set_ngpus (); 
  if (vftrace.cupti_state.n_devices > 0) {
      CUpti_SubscriberHandle subscriber; 
      cudaError_t ce;
      ce = cuptiSubscribe(&subscriber, 
                          (CUpti_CallbackFunc)vftr_cupti_event_callback,
                          NULL);
      if (ce != cudaSuccess) return ce;
      ce = cuptiEnableDomain(1, subscriber, CUPTI_CB_DOMAIN_RUNTIME_API);
      return ce;
  }
  return cudaSuccess;
}
