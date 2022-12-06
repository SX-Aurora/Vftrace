#include <stdlib.h>
#include <stddef.h>

#include <string.h>

#include <veda.h>

#include "stack_types.h"
#include "thread_types.h"
#include "threadstack_types.h"
#include "vftrace_state.h"
#include "stacks.h"
#include "threads.h"
#include "threadstacks.h"
#include "profiling.h"
#include "vedaprofiling_types.h"
#include "vedaprofiling.h"

#include "timer.h"
#include "veda_regions.h"

typedef struct {
   long long start_time;
   int threadID;
   int stackID;
   size_t bytes;
   long long overhead_time;
} user_data_t;

static const char *callbackname = "vedaMemcpyHtoD";
static const char *callbacknameAsync = "vedaMemcpyHtoDAsync";

void vftr_veda_callback_mem_cpy_htod_enter(VEDAprofiler_data* data) {
//printf("reqid=%d\n", data->req_id);
   long long tstart_callback = vftr_get_runtime_nsec();

   // Cast data->type to the callback specific struct
   VEDAprofiler_vedaMemcpy *MemcpyData;
   MemcpyData = (VEDAprofiler_vedaMemcpy*) &(data->type);

   // Start the veda region to add to the stacktree
   vftr_veda_region_begin(callbackname);
   // Obtain stack and thread ID for the called function
   // in order to pass it to the exit callback
   // to complete the function profiling
   thread_t *my_thread = vftr_get_my_thread(&(vftrace.process.threadtree));
   threadstack_t *my_threadstack = vftr_get_my_threadstack(my_thread);
   int stackID = my_threadstack->stackID;
   int threadID = my_thread->threadID;

   // Store collect data that needs to be passed to
   // the mem_cpy_exit callback
   // This includes the starting timestamp
   // Stack and thread ID of the called function
   user_data_t *user_data = (user_data_t*) malloc(sizeof(user_data_t));
   user_data->threadID = threadID;
   user_data->stackID = stackID;
   user_data->bytes = MemcpyData->bytes;
   user_data->start_time = vftr_get_runtime_nsec();
   data->user_data = (void*) user_data;

   user_data->overhead_time = vftr_get_runtime_nsec() - tstart_callback;
}

void vftr_veda_callback_mem_cpy_htod_exit(VEDAprofiler_data* data) {
   vftr_veda_region_end(callbackname);
   long long tstart_callback = vftr_get_runtime_nsec();
   long long memcpy_end_time = tstart_callback;

   // get back user data
   user_data_t *user_data = (user_data_t*) data->user_data;
   long long runtime_usec = memcpy_end_time - user_data->start_time;

   // store profiling data in profile
   profile_t *my_prof = vftr_get_my_profile_from_ids(user_data->stackID,
                                                   user_data->threadID);
   vftr_accumulate_veda_profiling_overhead(&(my_prof->vedaprof), user_data->overhead_time);
   my_prof->vedaprof.total_time_nsec += runtime_usec;
   my_prof->vedaprof.ncalls ++;
   my_prof->vedaprof.HtoD_bytes += user_data->bytes;
   my_prof->vedaprof.acc_HtoD_bw += user_data->bytes/(runtime_usec*1.0e-6);
   free(user_data);
   long long tend_callback = vftr_get_runtime_nsec();
   vftr_accumulate_veda_profiling_overhead(&(my_prof->vedaprof),
                                           tend_callback-tstart_callback);
}

void vftr_veda_callback_mem_cpy_htod_asyn_enter(VEDAprofiler_data* data) {
   //printf("req_id = %d\n", data->req_id);
   long long tstart_callback = vftr_get_runtime_nsec();

   // Cast data->type to the callback specific struct
   VEDAprofiler_vedaMemcpy *MemcpyData;
   MemcpyData = (VEDAprofiler_vedaMemcpy*) &(data->type);

   // Start the veda region to add to the stacktree
   vftr_veda_region_begin(callbacknameAsync);

   // Obtain stack and thread ID for the called function
   // in order to pass it to the exit callback
   // to complete the function profiling
   thread_t *my_thread = vftr_get_my_thread(&(vftrace.process.threadtree));
   threadstack_t *my_threadstack = vftr_get_my_threadstack(my_thread);
   int stackID = my_threadstack->stackID;
   int threadID = my_thread->threadID;
   // End the region here, as it is an asynchronous call,
   // thus the runtime is independend ofthe copy time,
   // thus no reson to wait for the exit call
   vftr_veda_region_end(callbackname);

   // Store collect data that needs to be passed to
   // the mem_cpy_exit callback
   // This includes the starting timestamp
   // Stack and thread ID of the called function
   user_data_t *user_data = (user_data_t*) malloc(sizeof(user_data_t));
   user_data->threadID = threadID;
   user_data->stackID = stackID;
   user_data->bytes = MemcpyData->bytes;
   user_data->start_time = vftr_get_runtime_nsec();
   data->user_data = (void*) user_data;
   user_data->overhead_time = vftr_get_runtime_nsec() - tstart_callback;
}

void vftr_veda_callback_mem_cpy_htod_async_exit(VEDAprofiler_data* data) {
   long long tstart_callback = vftr_get_runtime_nsec();
   long long memcpy_end_time = tstart_callback;

   // get back user data
   user_data_t *user_data = (user_data_t*) data->user_data;
   long long runtime_usec = memcpy_end_time - user_data->start_time;

   // store profiling data in profile
   profile_t *my_prof = vftr_get_my_profile_from_ids(user_data->stackID,
                                                   user_data->threadID);
   vftr_accumulate_veda_profiling_overhead(&(my_prof->vedaprof), user_data->overhead_time);
   my_prof->vedaprof.total_time_nsec += runtime_usec;
   my_prof->vedaprof.ncalls ++;
   my_prof->vedaprof.HtoD_bytes += user_data->bytes;
   my_prof->vedaprof.acc_HtoD_bw += user_data->bytes/(runtime_usec*1.0e-6);
   free(user_data);
   long long tend_callback = vftr_get_runtime_nsec();
   vftr_accumulate_veda_profiling_overhead(&(my_prof->vedaprof),
                                           tend_callback-tstart_callback);
}
