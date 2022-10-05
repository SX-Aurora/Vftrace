#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cupti.h>

#include "cupti_state_types.h"
#include "vftrace_state.h"
#include "callbacks.h"

int cupti_initialize () {
  int n_devices;
  cudaError_t ce;
  ce = cudaGetDeviceCount(&n_devices);
  if (ce != cudaSuccess) {
      vftrace.cupti_state.n_devices = 0;
      vftrace.cupti_state.event_buffer = NULL;
      return -1;
  } else {
      vftrace.cupti_state.n_devices = n_devices;
      vftrace.cupti_state.event_buffer = NULL;

  
      CUpti_SubscriberHandle subscriber; 
      ce = cuptiSubscribe(&subscriber, 
                          (CUpti_CallbackFunc)cupti_event_callback,
                          vftrace.cupti_state.event_buffer);
      ce = cuptiEnableDomain(1, subscriber, CUPTI_CB_DOMAIN_RUNTIME_API);
      return 0;
  }
}
