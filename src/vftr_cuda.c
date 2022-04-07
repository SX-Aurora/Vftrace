#include <stdio.h>
#include <stdbool.h>
#include <cupti.h>

#include "vftr_cuda.h"

// Structure to hold data collected by callback
//typedef struct RuntimeApiTrace_internal_st {
//  const char *functionName;
//  uint64_t startTimestamp;
//  uint64_t endTimestamp;
//  uint64_t t_acc_compute;
//  uint64_t t_acc_memcpy;
//  size_t memcpy_bytes;
//  enum cudaMemcpyKind memcpy_kind;
//} RuntimeApiTrace_internal_t;

enum launchOrder{ MEMCPY_H2D1, MEMCPY_H2D2, MEMCPY_D2H, KERNEL, THREAD_SYNC, LAUNCH_LAST};
enum traceElements {MEMCPY, ADD, MUL};

CUpti_SubscriberHandle subscriber;
//RuntimeApiTrace_t trace[LAUNCH_LAST];
//RuntimeApiTrace_t trace[3];
RuntimeApiTrace_t *traces;

kernel_list_t *kernel_list;

void CUPTIAPI
getTimestampCallback(void *userdata, CUpti_CallbackDomain domain,
                     CUpti_CallbackId cbid, const CUpti_CallbackData *cbInfo)
{
   if (!(cbid == CUPTI_RUNTIME_TRACE_CBID_cudaLaunch_v3020 ||
         cbid == CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000 ||
         cbid == CUPTI_RUNTIME_TRACE_CBID_cudaDeviceSynchronize_v3020 ||
         cbid == CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy_v3020)) return;

   uint64_t startTimestamp;
   uint64_t endTimestamp;
   static int memTransCount = 0;

   int use_id;
   if (cbid == CUPTI_RUNTIME_TRACE_CBID_cudaLaunch_v3020 || cbid == CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000) {
      use_id = 0;
   } else {
      use_id = 1;
   }

   const char *use_fun;
   if (use_id == 0) {
     use_fun = cbInfo->symbolName;
   } else {
     use_fun = cbInfo->functionName;
   } 
   //printf ("use_fun: %s\n", use_fun);

   //if (!strcmp(use_fun, "_Z10vector_addPfS_S_i")) {
   //  traceData += ADD;
   //} else if (!strcmp(use_fun, "_Z10vector_mulPfS_S_i")) {
   //  traceData += MUL;
   //}

   if (traces == NULL) {
      //printf ("Allocate TraceData!\n");
      traces = (RuntimeApiTrace_t*) malloc (sizeof(RuntimeApiTrace_t));
      traces->t_acc_compute = 0;
      traces->t_acc_memcpy = 0;
      traces->n_calls = 0;
      traces->functionName = use_fun;
      traces->next = NULL;
   }
   RuntimeApiTrace_t *this_trace = traces;
   
   this_trace = traces;
   bool found = false;
   while (true) {
      //printf ("Compare: %s %s\n", this_trace->functionName, use_fun);
      if (!strcmp(this_trace->functionName, use_fun)) {
         found = true;
         break;
      }
      if (this_trace->next == NULL) break;
      this_trace = this_trace->next;
   }
   //printf ("found %s: %d %d\n", use_fun, found, this_trace == NULL);

   if (!found) {
      //printf ("Allocate new: %s\n", use_fun);
      this_trace->next = (RuntimeApiTrace_t*) malloc (sizeof(RuntimeApiTrace_t));
      this_trace = this_trace->next;
      this_trace->functionName = use_fun;
      this_trace->t_acc_compute = 0;
      this_trace->t_acc_memcpy = 0;
      this_trace->n_calls = 0;
      this_trace->next = NULL;
   }

   if (cbInfo->callbackSite == CUPTI_API_ENTER) {

      if (cbid == CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy_v3020) {
         this_trace->memcpy_bytes = ((cudaMemcpy_v3020_params *)(cbInfo->functionParams))->count;
         //traces->memcpy_kind = ((cudaMemcpy_v3020_params *)(cbInfo->functionParams))->kind;
      }

      cuptiDeviceGetTimestamp(cbInfo->context, &startTimestamp); 
      this_trace->startTimestamp = startTimestamp;
   }

   if (cbInfo->callbackSite == CUPTI_API_EXIT) {
      cuptiDeviceGetTimestamp (cbInfo->context, &endTimestamp);
      this_trace->endTimestamp = endTimestamp;
      if (cbid == CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy_v3020) {
         this_trace->t_acc_memcpy += (this_trace->endTimestamp - this_trace->startTimestamp);
      } else {
         this_trace->t_acc_compute += (this_trace->endTimestamp - this_trace->startTimestamp);
      }
      this_trace->n_calls++;
      if (cbid == CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy_v3020) {
         memTransCount++;
      }
   }
}

void displayTimestamps (RuntimeApiTrace_t *trace) {
  //printf ("trace exists? %d\n", trace != NULL);
  while (trace != NULL) {
     printf ("%s: %lf us(compute), %lf us(memcpy) %d\n",
             trace->functionName,
             (double)trace->t_acc_compute / 1000,
             (double)trace->t_acc_memcpy / 1000,
             trace->n_calls);
     trace = trace->next;
  }
}

void setup_vftr_cuda () {
   printf ("Setting up vftr cuda!\n");
   //CUpti_SubscriberHandle subscriber;
   //RuntimeApiTrace_t trace[LAUNCH_LAST];
   //cuptiSubscribe(&subscriber, (CUpti_CallbackFunc)getTimestampCallback, &trace);
   cuptiSubscribe(&subscriber, (CUpti_CallbackFunc)getTimestampCallback, traces);
   cuptiEnableDomain(1, subscriber, CUPTI_CB_DOMAIN_RUNTIME_API);
   traces = NULL;
}

void vftr_cuda_flush_trace (RuntimeApiTrace_t **t) {
   *t = NULL;
   if (traces == NULL) return;

   RuntimeApiTrace_t *t_orig;
   RuntimeApiTrace_t *this_trace = traces;
   while (this_trace != NULL)  {
      if (*t == NULL) {
         *t = (RuntimeApiTrace_t*) malloc (sizeof(RuntimeApiTrace_t));
         t_orig = *t;
      } else {
         (*t)->next = (RuntimeApiTrace_t*) malloc (sizeof(RuntimeApiTrace_t));
         *t = (*t)->next;
      }
      (*t)->functionName = this_trace->functionName;
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
      RuntimeApiTrace_t *t_next = this_trace->next;
      free (this_trace);
      this_trace = t_next;
   }
   traces = NULL;
}

void final_vftr_cuda () {
   printf ("Unsubscribe CUPTI\n");
   //displayTimestamps(traces);
   cuptiUnsubscribe(subscriber);
}
