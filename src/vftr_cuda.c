#include <stdio.h>
#include <stdbool.h>
#include <cupti.h>

#include "vftr_cuda.h"

CUpti_SubscriberHandle subscriber;
cupti_trace_t *traces;

// This callback is evoked at the start and end of a CUDA function.
// We keep a list of trace elements, containing function names and runtime information,
// which is being filled until the list is flushed by Vftrace.

void CUPTIAPI vftr_cuda_callback_buffer(void *userdata, CUpti_CallbackDomain domain,
                                     CUpti_CallbackId cbid, const CUpti_CallbackData *cb_info) {
   // We only trace four kind of events: The launch of Cuda itself, kernel launches, synchronizations and memcpys.
   if (!(cbid == CUPTI_RUNTIME_TRACE_CBID_cudaLaunch_v3020 ||
         cbid == CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000 ||
         cbid == CUPTI_RUNTIME_TRACE_CBID_cudaDeviceSynchronize_v3020 ||
         cbid == CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy_v3020)) return;

   // For cuda functions, we use the symbolName. Otherwise, the correct call name is in functionName.
   const char *use_fun;
   if (cbid == CUPTI_RUNTIME_TRACE_CBID_cudaLaunch_v3020 || cbid == CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000) {
     use_fun = cb_info->symbolName;
   } else {
     use_fun = cb_info->functionName;
   } 

   // Init the trace list
   if (traces == NULL) {
      traces = (cupti_trace_t*) malloc (sizeof(cupti_trace_t));
      traces->t_acc_compute = 0;
      traces->t_acc_memcpy = 0;
      traces->n_calls = 0;
      traces->func_name = use_fun;
      traces->next = NULL;
   }
   cupti_trace_t *this_trace = traces;
   
   this_trace = traces;
   bool found = false;
   // Check if the function is currently in the list.
   while (true) {
      if (!strcmp(this_trace->func_name, use_fun)) {
         found = true;
         break;
      }
      if (this_trace->next == NULL) break;
      this_trace = this_trace->next;
   }

   if (!found) {
      this_trace->next = (cupti_trace_t*) malloc (sizeof(cupti_trace_t));
      this_trace = this_trace->next;
      this_trace->func_name = use_fun;
      this_trace->t_acc_compute = 0;
      this_trace->t_acc_memcpy = 0;
      this_trace->n_calls = 0;
      this_trace->next = NULL;
   }

   if (cb_info->callbackSite == CUPTI_API_ENTER) {

      if (cbid == CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy_v3020) {
         this_trace->memcpy_bytes = ((cudaMemcpy_v3020_params *)(cb_info->functionParams))->count;
      }

      uint64_t ts_start;
      cuptiDeviceGetTimestamp(cb_info->context, &(this_trace->ts_start)); 
   }

   if (cb_info->callbackSite == CUPTI_API_EXIT) {
      cuptiDeviceGetTimestamp (cb_info->context, &(this_trace->ts_end));
      if (cbid == CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy_v3020) {
         this_trace->t_acc_memcpy += (this_trace->ts_end - this_trace->ts_start);
      } else {
         this_trace->t_acc_compute += (this_trace->ts_end - this_trace->ts_start);
      }
      this_trace->n_calls++;
   }
}

void setup_vftr_cuda () {
   //printf ("Setting up vftr cuda!\n");
   cuptiSubscribe(&subscriber, (CUpti_CallbackFunc)vftr_cuda_callback_buffer, traces);
   cuptiEnableDomain(1, subscriber, CUPTI_CB_DOMAIN_RUNTIME_API);
   traces = NULL;
}

void vftr_cuda_flush_trace (cupti_trace_t **t) {
   *t = NULL;
   if (traces == NULL) return;

   cupti_trace_t *t_orig;
   cupti_trace_t *this_trace = traces;
   while (this_trace != NULL)  {
      if (*t == NULL) {
         *t = (cupti_trace_t*) malloc (sizeof(cupti_trace_t));
         t_orig = *t;
      } else {
         (*t)->next = (cupti_trace_t*) malloc (sizeof(cupti_trace_t));
         *t = (*t)->next;
      }
      (*t)->func_name = this_trace->func_name;
      (*t)->t_acc_compute = this_trace->t_acc_compute;
      (*t)->t_acc_memcpy = this_trace->t_acc_memcpy;
      (*t)->n_calls = this_trace->n_calls;
      (*t)->next = NULL; 
      this_trace = this_trace->next;
   }
   *t = t_orig;

   // Cleanup traces
   this_trace = traces;
   while (this_trace != NULL) {
      cupti_trace_t *t_next = this_trace->next;
      free (this_trace);
      this_trace = t_next;
   }
   traces = NULL;
}

void final_vftr_cuda () {
   //printf ("Unsubscribe CUPTI\n");
   cuptiUnsubscribe(subscriber);
}
