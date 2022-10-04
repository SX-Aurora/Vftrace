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
      return -1;
  }

  cupti_state_t cupti_state;
  cupti_state.n_devices = n_devices;
  cupti_state.event_buffer = NULL;

  
  CUpti_SubscriberHandle subscriber; 
  ce = cuptiSubscribe(&subscriber, (CUpti_CallbackFunc)cupti_event_callback, cupti_state.event_buffer);
  printf ("Subscribe: %d\n", ce == cudaSuccess);
  ce = cuptiEnableDomain(1, subscriber, CUPTI_CB_DOMAIN_RUNTIME_API);
  printf ("Enabled: %d\n", ce == cudaSuccess);
  vftrace.cupti_state = cupti_state;
  return 0;
}
