#include <cuda_runtime_api.h>
#include <cupti.h>

#include "thread_types.h"
#include "threadstack_types.h"
#include "stack_types.h"
#include "profiling_types.h"
#include "vftrace_state.h"
#include "timer.h"
#include "hashing.h"

#include "threads.h"
#include "threadstacks.h"
#include "profiling.h"
#include "callprofiling.h"

#include "cuptiprofiling.h"
#include "cupti_vftr_callbacks.h"

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

   // Note that this callback function is called everytime a CUDA (or OpenACC) function
   // is evoked. This means that CUDA functions for non-instrumented functions appear
   // with the same caller function. For example, a main function calling a cuBLAS DGEMM
   // will evoke several CUDA calls, of which none will call the default cyg-hook. In order
   // to not assign all CUDA calls to the same function, we need to identify them uniquely.
   // This cannot be done by the CBID alone, since different kernel launches will have the
   // same CBID. Therefore, for kernel launches, we compute a hash from the function name,
   // and add it to the CBID to create a pseudo-address. This is given to update the thread
   // stack as it would be for a normal address read from a symbol table. 

   const char *func_name;
   uint64_t pseudo_addr;
   if (cbid == CUPTI_RUNTIME_TRACE_CBID_cudaLaunch_v3020 || cbid == CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000) {
         func_name = cb_info->symbolName;
         pseudo_addr = vftr_jenkins_murmur_64_hash (strlen(func_name), (uint8_t*)func_name);
   } else {
         func_name = cb_info->functionName;
         pseudo_addr = 0;
   }
   pseudo_addr += cbid;

   my_threadstack = vftr_update_threadstack_region (my_threadstack, my_thread,
                                                    (uintptr_t)pseudo_addr, func_name,
                                                    &vftrace, false);

   stack_t *my_new_stack = vftrace.process.stacktree.stacks + my_threadstack->stackID;
   my_profile = vftr_get_my_profile(my_new_stack, my_thread);
   cuptiprofile_t *cuptiprof = &my_profile->cuptiprof;

   // We keep track of two timers: The default Vftrace call timer and the CUPTI event timer.
   // The latter one is started with cudaEventRecord. We therefore pass 0 to the accumulation
   // of the cupti profile. In the exit hook, the time difference between the start and stop event
   // is accumulated. The Vftrace timer and the CUPTI timer should be more or less equal.
   //
   // cb_info contains the same information regarding copied bytes for Memcpy events. Here, we
   // accumulate zero. The memory information is retrieved in the exit hook.
   cudaEventRecord(cuptiprof->start, 0);
   long long region_begin_time = vftr_get_runtime_nsec();
   vftr_accumulate_cuptiprofiling (cuptiprof, cbid, 1, 0, CUPTI_NOCOPY, 0);
   vftr_accumulate_callprofiling (&(my_profile->callprof), 1, -region_begin_time);
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

// The default CUPTI callback handle. It vetoes some CBIDs
// and then just calls the corresponding entry or exit region.
void CUPTIAPI vftr_cupti_event_callback (void *userdata,
                                         CUpti_CallbackDomain domain,
                                         CUpti_CallbackId cbid,
 					 const CUpti_CallbackData *cb_info) {
   // These functions are called in the profiling layer. We need to exclude them,
   // otherwise the callback will be called infinitely
   if (cbid == CUPTI_RUNTIME_TRACE_CBID_cudaEventRecord_v3020
    || cbid == CUPTI_RUNTIME_TRACE_CBID_cudaEventRecordWithFlags_v11010
    || cbid == CUPTI_RUNTIME_TRACE_CBID_cudaEventSynchronize_v3020
    || cbid == CUPTI_RUNTIME_TRACE_CBID_cudaEventElapsedTime_v3020
    || cbid == CUPTI_RUNTIME_TRACE_CBID_cudaEventCreate_v3020
    || cbid == CUPTI_RUNTIME_TRACE_CBID_cudaEventCreateWithFlags_v3020
    || cbid == CUPTI_RUNTIME_TRACE_CBID_cudaEventDestroy_v3020
    || cbid == CUPTI_RUNTIME_TRACE_CBID_cudaEventQuery_v3020) return;

   if (cb_info->callbackSite == CUPTI_API_ENTER) {
      vftr_cupti_region_begin (cbid, cb_info);
   } else if (cb_info->callbackSite == CUPTI_API_EXIT) {
      vftr_cupti_region_end (cbid, cb_info);
   }
}

