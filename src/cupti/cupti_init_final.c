#include <cuda_runtime_api.h>

#include "vftrace_state.h"
#include "callbacks.h"

void cupti_initialize () {
  int n_devices;
  cudaError_t ce;
  ce = cudaGetDeviceCount(&n_devices);
  if (ce != cudaSuccess) {
      vftrace.cupti_state.n_devices = 0;
  } else {
      vftrace.cupti_state.n_devices = n_devices;
  
      CUpti_SubscriberHandle subscriber; 
      ce = cuptiSubscribe(&subscriber, 
                          (CUpti_CallbackFunc)cupti_event_callback,
                          NULL);
      ce = cuptiEnableDomain(1, subscriber, CUPTI_CB_DOMAIN_RUNTIME_API);
  }
}
