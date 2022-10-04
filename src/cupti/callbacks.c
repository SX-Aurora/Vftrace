#include <cupti.h>
#include <cuda_runtime_api.h>
#include "vftrace_state.h"

#define CUPTI_RUNTIME_TRACE_CBID_cudaLaunch_v3020 13
#define CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000 211
#define CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy_v3020 31
#define CUPTI_RUNTIME_TRACE_CBID_cudaMemcpyAsync_v3020 41

void CUPTIAPI cupti_event_callback (void *userdata, CUpti_CallbackDomain domain,
                                    CUpti_CallbackId cbid, const CUpti_CallbackData *cb_info) {
   if (!(cbid == CUPTI_RUNTIME_TRACE_CBID_cudaLaunch_v3020 ||
         cbid == CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000 ||
         cbid == CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy_v3020 ||
         cbid == CUPTI_RUNTIME_TRACE_CBID_cudaMemcpyAsync_v3020)) return;

   printf ("Event callback called!\n");
   char *use_fun;
   if (cbid == CUPTI_RUNTIME_TRACE_CBID_cudaLaunch_v3020 || cbid == CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000) {
      use_fun = cb_info->symbolName;
   } else {
      use_fun = cb_info->functionName;
   }

   if (vftrace.cupti_state.event_buffer == NULL) {
      vftrace.cupti_state.event_buffer = (cupti_event_list_t*)malloc(sizeof(cupti_event_list_t)); 
      vftrace.cupti_state.event_buffer->func_name = use_fun;
      vftrace.cupti_state.event_buffer->cbid = cbid;
      vftrace.cupti_state.event_buffer->n_calls = 0;
      vftrace.cupti_state.event_buffer->memcpy_bytes = 0;
      vftrace.cupti_state.event_buffer->t_acc[T_CUDA_COMP] = 0;
      vftrace.cupti_state.event_buffer->t_acc[T_CUDA_MEMCP] = 0;
      cudaEventCreate (&(vftrace.cupti_state.event_buffer->start));
      cudaEventCreate (&(vftrace.cupti_state.event_buffer->stop));
      vftrace.cupti_state.event_buffer->next = NULL;
   }

   cupti_event_list_t *this_event = vftrace.cupti_state.event_buffer;
   while (this_event != NULL && strcmp(this_event->func_name, use_fun)) {
      this_event = this_event->next;
   }

   if (this_event == NULL) {
      this_event = (cupti_event_list_t*) malloc (sizeof(cupti_event_list_t));
      this_event->func_name = use_fun;
      this_event->cbid = cbid;
      this_event->n_calls = 0;
      this_event->memcpy_bytes = 0;
      this_event->t_acc[T_CUDA_COMP] = 0;
      this_event->t_acc[T_CUDA_MEMCP] = 0;
      cudaEventCreate (&(this_event->start));
      cudaEventCreate (&(this_event->stop));
      this_event->next = NULL;
   }

   if (cb_info->callbackSite == CUPTI_API_ENTER) {
      cudaEventRecord(this_event->start, 0);
   } else if (cb_info->callbackSite == CUPTI_API_EXIT) {
      cudaEventRecord(this_event->stop, 0);
      cudaEventSynchronize(this_event->stop);
      float t;
      cudaEventElapsedTime(&t, this_event->start, this_event->stop);
      int type = (cbid == CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy_v3020
               || cbid == CUPTI_RUNTIME_TRACE_CBID_cudaMemcpyAsync_v3020) ? T_CUDA_MEMCP : T_CUDA_COMP;
      this_event->t_acc[type] += t;
      this_event->n_calls++;
      //if (cbid == CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy_v3020) {
      //   this_event->memcpy_bytes += ((cudaMemcpy_v3020_params *)(cb_info->functionParams))->count;
      //} else if (cbid == CUPTI_RUNTIME_TRACE_CBID_cudaMemcpyAsync_v3020) {
      //   this_event->memcpy_bytes += ((cudaMemcpyAsync_v3020_params *)(cb_info->functionParams))->count;
      //}
   }
}

