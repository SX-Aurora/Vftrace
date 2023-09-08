#include <cuda_runtime_api.h>
#include <cupti.h>

#include "thread_types.h"
#include "threadstack_types.h"
#include "stack_types.h"
#include "profiling_types.h"
#include "vftrace_state.h"
#include "timer.h"
#include "hashing.h"
#include "self_profile.h"

#include "threads.h"
#include "threadstacks.h"
#include "profiling.h"
#include "callprofiling.h"

#include "cudaprofiling.h"
#include "cupti_vftr_callbacks.h"

void vftr_get_cuda_memory_info (CUpti_CallbackId cbid, const CUpti_CallbackData *cb_info,
                           int *mem_dir, size_t *copied_bytes) {

   if (cbid != CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy_v3020 &&
       cbid != CUPTI_RUNTIME_TRACE_CBID_cudaMemcpyAsync_v3020) {
          *mem_dir = CUDA_NOCOPY;
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
      *mem_dir = kind == cudaMemcpyHostToDevice ? CUDA_COPY_IN : CUDA_COPY_OUT;
   }
}

void vftr_cuda_region_begin (int cbid, const CUpti_CallbackData *cb_info) {
   SELF_PROFILE_START_FUNCTION;
   long long region_entry_time_begin = vftr_get_runtime_nsec();
   thread_t *my_thread = vftr_get_my_thread(&(vftrace.process.threadtree));
   threadstack_t *my_threadstack = vftr_get_my_threadstack(my_thread);

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

   vftr_stack_t *my_new_stack = vftrace.process.stacktree.stacks + my_threadstack->stackID;
   profile_t *my_profile = vftr_get_my_profile(my_new_stack, my_thread);
   cudaprofile_t *cudaprof = &my_profile->cudaprof;

   // We fill the callprofile and the cudaprofile with the same time measurements obtained
   // from CUpti. When using the Vftrace timer for the callprofile, in any case the
   // overhead of the CUpti calls is included in some way. This gives a wrong impression
   // of the time spent. That's why below we pass 0 to accumulate_callprofiling, as the
   // exit function will supply the time difference.

   // cb_info contains the same information regarding copied bytes for Memcpy events. Here, we
   // accumulate zero. The memory information is retrieved in the exit hook.
   cudaEventRecord(cudaprof->start, 0);
   
   int mem_dir;
   size_t copied_bytes;
   vftr_get_cuda_memory_info (cbid, cb_info, &mem_dir, &copied_bytes);

   vftr_accumulate_cudaprofiling (cudaprof, cbid, 1, 0, mem_dir, 0);
   vftr_accumulate_callprofiling (&(my_profile->callprof), 1, 0);
   
   vftr_accumulate_cudaprofiling_overhead (&(my_profile->cudaprof),
                  vftr_get_runtime_nsec() - region_entry_time_begin);
   SELF_PROFILE_END_FUNCTION;
}

void vftr_cuda_region_end (int cbid, const CUpti_CallbackData *cb_info) {
   SELF_PROFILE_START_FUNCTION;
   (void)cb_info;
   long long region_exit_time_begin = vftr_get_runtime_nsec();

   thread_t *my_thread = vftr_get_my_thread(&(vftrace.process.threadtree));
   threadstack_t *my_threadstack = vftr_get_my_threadstack(my_thread);
   vftr_stack_t *my_stack = vftrace.process.stacktree.stacks + my_threadstack->stackID;
   profile_t *my_profile = vftr_get_my_profile(my_stack, my_thread);
   cudaprofile_t *cudaprof = &my_profile->cudaprof;

   cudaEventRecord(cudaprof->stop, 0);
   cudaEventSynchronize(cudaprof->stop);
   float t;
   cudaEventElapsedTime(&t, cudaprof->start, cudaprof->stop);

   int mem_dir;
   size_t copied_bytes;
   vftr_get_cuda_memory_info (cbid, cb_info, &mem_dir, &copied_bytes);

   vftr_accumulate_cudaprofiling(cudaprof, cbid, 0, t, mem_dir, copied_bytes);
   // Convert ms -> ns
   vftr_accumulate_callprofiling(&(my_profile->callprof), 0, (long long)(t * 1000000));
   (void)vftr_threadstack_pop(&(my_thread->stacklist));
   vftr_accumulate_cudaprofiling_overhead (&(my_profile->cudaprof),
                   vftr_get_runtime_nsec() - region_exit_time_begin);
   SELF_PROFILE_END_FUNCTION;
}

// The default CUpti callback handle. It vetoes some CBIDs
// and then just calls the corresponding entry or exit region.
void CUPTIAPI vftr_cupti_event_callback (void *userdata,
                                         CUpti_CallbackDomain domain,
                                         CUpti_CallbackId cbid,
 					 const CUpti_CallbackData *cb_info) {
   // These functions are called in the profiling layer. We need to exclude them,
   // otherwise the callback will be called infinitely
   switch (cbid) {
      case CUPTI_RUNTIME_TRACE_CBID_cudaEventRecord_v3020:
      case CUPTI_RUNTIME_TRACE_CBID_cudaEventRecordWithFlags_v11010:
      case CUPTI_RUNTIME_TRACE_CBID_cudaEventSynchronize_v3020:
      case CUPTI_RUNTIME_TRACE_CBID_cudaEventElapsedTime_v3020:
      case CUPTI_RUNTIME_TRACE_CBID_cudaEventCreate_v3020:
      case CUPTI_RUNTIME_TRACE_CBID_cudaEventCreateWithFlags_v3020:
      case CUPTI_RUNTIME_TRACE_CBID_cudaEventDestroy_v3020:
      case CUPTI_RUNTIME_TRACE_CBID_cudaEventQuery_v3020:
      case CUPTI_RUNTIME_TRACE_CBID_cudaGetDeviceProperties_v3020:
      case CUPTI_RUNTIME_TRACE_CBID_cudaGetDeviceCount_v3020:
      case CUPTI_RUNTIME_TRACE_CBID_cudaGetDeviceProperties_v2_v12000:
         return;
   }

   if (cb_info->callbackSite == CUPTI_API_ENTER) {
      vftr_cuda_region_begin (cbid, cb_info);
   } else if (cb_info->callbackSite == CUPTI_API_EXIT) {
      vftr_cuda_region_end (cbid, cb_info);
   }
}

