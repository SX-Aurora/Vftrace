#include <cupti.h>
#include <cuda_runtime_api.h>

#include "thread_types.h"
#include "threadstack_types.h"
#include "stack_types.h"
#include "profiling_types.h"
#include "vftrace_state.h"

#include "threads.h"
#include "threadstacks.h"
#include "profiling.h"

#include "cupti_event_list.h"

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

   char *use_fun;
   if (cbid == CUPTI_RUNTIME_TRACE_CBID_cudaLaunch_v3020 || cbid == CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000) {
      use_fun = cb_info->symbolName;
   } else {
      use_fun = cb_info->functionName;
   }

   thread_t *my_thread = vftr_get_my_thread(&(vftrace.process.threadtree));
   threadstack_t *my_threadstack = vftr_get_my_threadstack(my_thread);
   stack_t *my_stack = vftrace.process.stacktree.stacks+my_threadstack->stackID;
   profile_t *my_profile = vftr_get_my_profile(my_stack, my_thread);
   cuptiprofile_t *cuptiprof = &(my_profile->cuptiprof);

   if (cb_info->callbackSite == CUPTI_API_ENTER) {
      cudaEventRecord(cuptiprof->start, 0);
   } else if (cb_info->callbackSite == CUPTI_API_EXIT) {
      cudaEventRecord(cuptiprof->stop, 0);
      cudaEventSynchronize(cuptiprof->stop);
      float t;
      cudaEventElapsedTime(&t, cuptiprof->start, cuptiprof->stop);
      int type = (cbid == CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy_v3020
               || cbid == CUPTI_RUNTIME_TRACE_CBID_cudaMemcpyAsync_v3020) ? T_CUDA_MEMCP : T_CUDA_COMP;
      if (cuptiprof->events == NULL) {
          cuptiprof->events = new_cupti_event (use_fun, cbid, t, 0);
      } else {
          cupti_event_list_t *this_event = cuptiprof->events;
          while (this_event->next != NULL && strcmp(this_event->func_name, use_fun)) {
              this_event = this_event->next; 
          }
          if (this_event == NULL) {
              this_event = new_cupti_event (use_fun, cbid, t, 0);
          } else {
              acc_cupti_event (this_event, t, 0);
          }
      }
   }
}

