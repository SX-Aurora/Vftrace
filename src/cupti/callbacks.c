#include <cuda_runtime_api.h>
#include <cupti.h>

#include "thread_types.h"
#include "threadstack_types.h"
#include "stack_types.h"
#include "profiling_types.h"
#include "vftrace_state.h"
#include "timer.h"

#include "threads.h"
#include "threadstacks.h"
#include "profiling.h"
#include "callprofiling.h"

#include "cuptiprofiling.h"
#include "callbacks.h"

void vftr_get_cupti_memory_info (CUpti_CallbackId cbid, const CUpti_CallbackData *cb_info,
                           int *mem_dir, size_t *copied_bytes) {

   if (cbid != CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy_v3020 &&
       cbid != CUPTI_RUNTIME_TRACE_CBID_cudaMemcpyAsync_v3020) {
          *mem_dir = CUPTI_NOCOPY;
          *copied_bytes = 0;
   } else {
      enum cudaMemcpyKind kind;
      if (cbid == CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy_v3020) {
         *copied_bytes = (size_t)((cudaMemcpy_v3020_params*)(cb_info->functionParams))->count;
         kind = ((cudaMemcpy_v3020_params*)(cb_info->functionParams))->kind;
      } else {
         *copied_bytes = (size_t)((cudaMemcpyAsync_v3020_params*)(cb_info->functionParams))->count;
         kind = ((cudaMemcpyAsync_v3020_params*)(cb_info->functionParams))->kind;
      }
      *mem_dir = kind == cudaMemcpyHostToDevice ? CUPTI_COPY_IN : CUPTI_COPY_OUT;
   }
}

void vftr_cupti_region_begin (int cbid, const CUpti_CallbackData *cb_info) {
   thread_t *my_thread = vftr_get_my_thread(&(vftrace.process.threadtree));
   threadstack_t *my_threadstack = vftr_get_my_threadstack(my_thread);
   stack_t *my_stack = vftrace.process.stacktree.stacks+my_threadstack->stackID;
   profile_t *my_profile = vftr_get_my_profile(my_stack, my_thread);   

   char *func_name;
   if (cbid == CUPTI_RUNTIME_TRACE_CBID_cudaLaunch_v3020 || cbid == CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000) {
         func_name = cb_info->symbolName;
   } else {
         func_name = cb_info->functionName;
   }

   // Cuda calls are not recursive
   my_threadstack = vftr_update_threadstack_region (my_threadstack, my_thread,
                                                    (void*)cb_info, func_name,
                                                    cupti_region, &vftrace, false);

   stack_t *my_new_stack = vftrace.process.stacktree.stacks + my_threadstack->stackID;
   my_profile = vftr_get_my_profile(my_new_stack, my_thread);
   cuptiprofile_t *cuptiprof = &my_profile->cuptiprof;
   long long region_begin_time = vftr_get_runtime_nsec();
   vftr_accumulate_cuptiprofiling (cuptiprof, cbid, 1, 0, CUPTI_NOCOPY, 0);
   vftr_accumulate_callprofiling (&(my_profile->callprof), 1, -region_begin_time);

   cudaEventRecord(cuptiprof->start, 0);
}

void vftr_cupti_region_end (int cbid, const CUpti_CallbackData *cb_info) {
   (void)cb_info;
   long long region_end_time = vftr_get_runtime_nsec();

   thread_t *my_thread = vftr_get_my_thread(&(vftrace.process.threadtree));
   threadstack_t *my_threadstack = vftr_get_my_threadstack(my_thread);
   stack_t *my_stack = vftrace.process.stacktree.stacks+my_threadstack->stackID;
   profile_t *my_profile = vftr_get_my_profile(my_stack, my_thread);
   cuptiprofile_t *cuptiprof = &my_profile->cuptiprof;

   cudaEventRecord(cuptiprof->stop, 0);
   cudaEventSynchronize(cuptiprof->stop);
   float t;
   cudaEventElapsedTime(&t, cuptiprof->start, cuptiprof->stop);

   int mem_dir;
   size_t copied_bytes;
   vftr_get_cupti_memory_info (cbid, cb_info, &mem_dir, &copied_bytes);

   vftr_accumulate_cuptiprofiling(cuptiprof, cbid, 0, t, mem_dir, copied_bytes);
   vftr_accumulate_callprofiling(&(my_profile->callprof), 0, region_end_time);
   (void)vftr_threadstack_pop(&(my_thread->stacklist));

}

void CUPTIAPI vftr_cupti_event_callback (void *userdata,
                                         CUpti_CallbackDomain domain,
                                         CUpti_CallbackId cbid,
 					 const CUpti_CallbackData *cb_info) {
    // These functions are called in the profiling layer. We need to exclude them,
   // otherwise the callback will be called infinitely
   if (cbid == CUPTI_RUNTIME_TRACE_CBID_cudaEventRecord_v3020
    || cbid == CUPTI_RUNTIME_TRACE_CBID_cudaEventSynchronize_v3020
    || cbid == CUPTI_RUNTIME_TRACE_CBID_cudaEventElapsedTime_v3020
    || cbid == CUPTI_RUNTIME_TRACE_CBID_cudaEventCreate_v3020
    || cbid == CUPTI_RUNTIME_TRACE_CBID_cudaGetDeviceProperties_v3020) return;

   if (cb_info->callbackSite == CUPTI_API_ENTER) {
      vftr_cupti_region_begin (cbid, cb_info);
   } else if (cb_info->callbackSite == CUPTI_API_EXIT) {
      vftr_cupti_region_end (cbid, cb_info);
   }
}

