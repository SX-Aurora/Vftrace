#include <stdbool.h>
#include <cupti_runtime_cbid.h>

#include "cupti_event_types.h"
#include "vftrace_state.h"

cupti_event_list_t *new_cupti_event (char *func_name, int cbid, float t_ms, uint64_t memcpy_bytes) {
   cupti_event_list_t *new_event = (cupti_event_list_t*)malloc(sizeof(cupti_event_list_t));   
   new_event->func_name = func_name;
   new_event->cbid = cbid;
   new_event->n_calls = 1;
   new_event->t_ms = t_ms;
   new_event->memcpy_bytes = memcpy_bytes;
   new_event->next = NULL;
   return new_event;
}

void acc_cupti_event (cupti_event_list_t *event, float t_ms, uint64_t memcpy_bytes) {
   event->n_calls++;
   event->t_ms += t_ms;
   event->memcpy_bytes += memcpy_bytes;
}

bool cupti_event_belongs_to_class (cupti_event_list_t *event, int cbid_class) {
   bool is_compute =  event->cbid == CUPTI_RUNTIME_TRACE_CBID_cudaLaunch_v3020
                   || event->cbid == CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000;
   bool is_memcpy = event->cbid == CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy_v3020
                 || event->cbid == CUPTI_RUNTIME_TRACE_CBID_cudaMemcpyAsync_v3020;
   bool is_other = !is_compute && !is_memcpy;

   switch (cbid_class) {
      case T_CUPTI_COMP:
         return is_compute;
      case T_CUPTI_MEMCP:
         return is_memcpy;
      case T_CUPTI_OTHER:
         return is_other;
      default:
         return false; 
   }
}


