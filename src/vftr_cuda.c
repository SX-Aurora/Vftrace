#include <stdio.h>
#include <stdbool.h>
#ifdef _CUPTI_AVAIL
#include <cupti.h>
#endif

#include "vftr_environment.h"
#include "vftr_cuda.h"


int vftr_n_cuda_devices;
int vftr_registered_cbids[406];

/**********************************************************************/

bool vftr_profile_cuda () {
   return vftr_n_cuda_devices > 0 && !vftr_environment.ignore_cuda->value;
}

/**********************************************************************/

#ifdef _CUPTI_AVAIL
CUpti_SubscriberHandle subscriber;
cuda_event_list_internal_t *events;
struct cudaDeviceProp vftr_cuda_properties;


// This callback is evoked at the start and end of a CUDA function.
// We keep a list of trace elements, containing function names and runtime information,
// which is being filled until the list is flushed by Vftrace.
void CUPTIAPI vftr_cuda_callback_events(void *userdata, CUpti_CallbackDomain domain,
                                        CUpti_CallbackId cbid, const CUpti_CallbackData *cb_info) {
   vftr_registered_cbids[cbid]++;
   // We only trace four kind of events: The launch of Cuda itself, kernel launches, synchronizations and memcpys.
   if (!(cbid == CUPTI_RUNTIME_TRACE_CBID_cudaLaunch_v3020 ||
         cbid == CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000 ||
         cbid == CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy_v3020 ||
         cbid == CUPTI_RUNTIME_TRACE_CBID_cudaMemcpyAsync_v3020)) return;

   // For cuda functions, we use the symbolName. Otherwise, the correct call name is in functionName.
   char *use_fun;
   if (cbid == CUPTI_RUNTIME_TRACE_CBID_cudaLaunch_v3020 || cbid == CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000) {
     use_fun = cb_info->symbolName;
   } else {
     use_fun = cb_info->functionName;
   } 

   // Init event list
   if (events == NULL) {
     events = (cuda_event_list_internal_t*) malloc (sizeof(cuda_event_list_internal_t));
     events->func_name = use_fun;
     events->cbid = cbid;
     cudaEventCreate (&(events->start));
     cudaEventCreate (&(events->stop));
     events->t_acc[T_CUDA_COMP] = 0;
     events->t_acc[T_CUDA_MEMCP] = 0;
     events->n_calls = 0;
     events->memcpy_bytes = 0;
     events->next = NULL;
   }
   
   cuda_event_list_internal_t *this_event = events;

   while (this_event != NULL && strcmp(this_event->func_name, use_fun)) {
      this_event = this_event->next;
   }
   if (this_event == NULL) {
      this_event = (cuda_event_list_internal_t*) malloc (sizeof(cuda_event_list_internal_t));
      this_event->func_name = use_fun;
      this_event->cbid = cbid;
      cudaEventCreate (&(this_event->start));
      cudaEventCreate (&(this_event->stop));
      this_event->n_calls = 0;
      this_event->memcpy_bytes = 0;
      this_event->t_acc[T_CUDA_COMP] = 0;
      this_event->t_acc[T_CUDA_MEMCP] = 0;
      this_event->next = NULL;
   }
   
   if (cb_info->callbackSite == CUPTI_API_ENTER) {
     cudaEventRecord(this_event->start, 0); 
   } else if (cb_info->callbackSite == CUPTI_API_EXIT) {
     cudaEventRecord(this_event->stop, 0);
     cudaEventSynchronize(this_event->stop);
     float t;
     cudaEventElapsedTime(&t, this_event->start, this_event->stop);
     int type = (cbid == CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy_v3020 || cbid == CUPTI_RUNTIME_TRACE_CBID_cudaMemcpyAsync_v3020) ? T_CUDA_MEMCP : T_CUDA_COMP;
     this_event->t_acc[type] += t;
     this_event->n_calls++;
     if (cbid == CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy_v3020) {
        this_event->memcpy_bytes += ((cudaMemcpy_v3020_params *)(cb_info->functionParams))->count;
     } else if (cbid == CUPTI_RUNTIME_TRACE_CBID_cudaMemcpyAsync_v3020) {
        this_event->memcpy_bytes += ((cudaMemcpyAsync_v3020_params *)(cb_info->functionParams))->count;
     }

   }
}

#endif

/**********************************************************************/

void vftr_setup_cuda () {
#if defined(_CUPTI_AVAIL)
   cudaError_t ce = cudaGetDeviceCount(&vftr_n_cuda_devices);
   if (ce != cudaSuccess) {
       vftr_n_cuda_devices = 0; 
   } else {
       cudaGetDeviceProperties (&vftr_cuda_properties, 0);
   }
   events = NULL;
   cuptiSubscribe(&subscriber, (CUpti_CallbackFunc)vftr_cuda_callback_events, events);
   cuptiEnableDomain(1, subscriber, CUPTI_CB_DOMAIN_RUNTIME_API);

   for (int i = 0; i < 406; i++) {
      vftr_registered_cbids[i] = 0;
   }
#else
   vftr_n_cuda_devices = 0;
#endif
}

/**********************************************************************/

void vftr_cuda_flush_events (cuda_event_list_t **t) {
  *t = NULL;
#ifdef _CUPTI_AVAIL
  if (events == NULL) return;
  
  cuda_event_list_internal_t *this_event = events;
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
     (*t)->cbid = this_event->cbid;
     (*t)->memcpy_bytes = this_event->memcpy_bytes;
     (*t)->next = NULL;
     this_event = this_event->next; 
  }
  *t = t_orig;

  this_event = events;
  while (this_event != NULL) {
     cuda_event_list_internal_t *t_next = this_event->next;
     free (this_event);
     this_event = t_next;
  } 
  events = NULL;
#endif
} 

/**********************************************************************/

void vftr_print_registered_cbids (FILE *fp) {
   fprintf (fp, "Registered CBIDs: \n");
   for (int i = 0; i < 406; i++) {
      if (vftr_registered_cbids[i] > 0) fprintf (fp, "%d: %d\n", i, vftr_registered_cbids[i]); 
   }
   fprintf (fp, "\n");
}

/**********************************************************************/

void vftr_final_cuda () {
#ifdef _CUPTI_AVAIL
   cuptiUnsubscribe(subscriber);
#endif
}

/**********************************************************************/
