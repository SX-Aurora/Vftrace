#include <stdio.h>
#include <stdbool.h>
#include <cupti.h>

#include "vftr_cuda.h"

CUpti_SubscriberHandle subscriber;
cuda_event_list_t *events;

// This callback is evoked at the start and end of a CUDA function.
// We keep a list of trace elements, containing function names and runtime information,
// which is being filled until the list is flushed by Vftrace.
void CUPTIAPI vftr_cuda_callback_events(void *userdata, CUpti_CallbackDomain domain,
                                        CUpti_CallbackId cbid, const CUpti_CallbackData *cb_info) {
   // We only trace four kind of events: The launch of Cuda itself, kernel launches, synchronizations and memcpys.
   if (!(cbid == CUPTI_RUNTIME_TRACE_CBID_cudaLaunch_v3020 ||
         cbid == CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000 ||
         cbid == CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy_v3020)) return;

   // For cuda functions, we use the symbolName. Otherwise, the correct call name is in functionName.
   //const char *use_fun;
   char *use_fun;
   if (cbid == CUPTI_RUNTIME_TRACE_CBID_cudaLaunch_v3020 || cbid == CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000) {
     use_fun = cb_info->symbolName;
   } else {
     use_fun = cb_info->functionName;
   } 

   // Init event list
   if (events == NULL) {
     events = (cuda_event_list_t*) malloc (sizeof(cuda_event_list_t));
     events->func_name = use_fun;
     cudaEventCreate (&(events->start));
     cudaEventCreate (&(events->stop));
     events->t_acc[T_CUDA_COMP] = 0;
     events->t_acc[T_CUDA_MEMCP] = 0;
     events->n_calls = 0;
     events->memcpy_bytes = 0;
     events->next = NULL;
   }
   
   cuda_event_list_t *this_event = events;
   //printf ("In CUPTI Callback!\n");

   while (this_event != NULL && strcmp(this_event->func_name, use_fun)) {
      this_event = this_event->next;
   }
   if (this_event == NULL) {
      this_event = (cuda_event_list_t*) malloc (sizeof(cuda_event_list_t));
      this_event->func_name = use_fun;
      cudaEventCreate (&(this_event->start));
      cudaEventCreate (&(this_event->stop));
      this_event->n_calls = 0;
      this_event->memcpy_bytes = 0;
      this_event->t_acc[T_CUDA_COMP] = 0;
      this_event->t_acc[T_CUDA_MEMCP] = 0;
      this_event->next = NULL;
   //} else {
   //   this_event = this_event->next;
   }
   
   if (cb_info->callbackSite == CUPTI_API_ENTER) {
     cudaEventRecord(this_event->start, 0); 
   } else if (cb_info->callbackSite == CUPTI_API_EXIT) {
     cudaEventRecord(this_event->stop, 0);
     cudaEventSynchronize(this_event->stop);
     float t;
     cudaEventElapsedTime(&t, this_event->start, this_event->stop);
     if (cbid == CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy_v3020) {
        this_event->t_acc_memcpy += t;
     } else {
        this_event->t_acc_compute += t;
     }
   }
   //printf ("Out CUPTI Callback!\n");

}

void setup_vftr_cuda () {
   cuptiSubscribe(&subscriber, (CUpti_CallbackFunc)vftr_cuda_callback_events, events);
   cuptiEnableDomain(1, subscriber, CUPTI_CB_DOMAIN_RUNTIME_API);
   events = NULL;
}

void vftr_cuda_flush_events (cuda_event_list_t **t) {
  *t = NULL;
  if (events == NULL) return;
  
  cuda_event_list_t *this_event = events;
  cuda_event_list_t *t_orig;
  while (this_event != NULL) {
     if (*t == NULL) {
        *t = (cuda_event_list_t*) malloc (sizeof(cuda_event_list_t));
        t_orig = *t;
     } else {
        (*t)->next = (cuda_event_list_t*) malloc (sizeof(cuda_event_list_t));
        *t = (*t)->next;
     }   
     (*t)->func_name = this_event->func_name;
     (*t)->t_acc[T_CUDA_COMP] = this_event->t_acc[T_CUDA_COMP];
     (*t)->t_acc[T_CUDA_MEMCP] = this_event->t_acc[T_CUDA_MEMCP];
     (*t)->n_calls = this_event->n_calls;
     (*t)->next = NULL;
     this_event = this_event->next; 
  }
  *t = t_orig;

  this_event = events;
  while (this_event != NULL) {
     cuda_event_list_t *t_next = this_event->next;
     free (this_event);
     this_event = t_next;
  } 
  events = NULL;
} 

void final_vftr_cuda () {
   cuptiUnsubscribe(subscriber);
}
