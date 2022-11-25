#include <stdlib.h>

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

#include "timer.h"
#include "veda_regions.h"

typedef struct {
   long long start_time;
   int threadID;
   int stackID;
} user_data_t;

void vftr_veda_callback_launch_kernel_enter(VEDAprofiler_data* data) {
   long long tstart_callback = vftr_get_runtime_nsec();

   VEDAprofiler_vedaLaunchKernel *LaunchKernelData;
   LaunchKernelData = (VEDAprofiler_vedaLaunchKernel*) &(data->type);
   char *launchname = "vedaLaunchKernel";
   int launchnamelen = strlen(launchname);
   char *kernelname = LaunchKernelData->kernel;
   int kernelnamelen = strlen(kernelname);
   int totallen = launchnamelen +
                  kernelnamelen +
                  2 + // +2 for braces
                  1; // +1 for null terminator
   char *combinedname = (char*) malloc(totallen*sizeof(char));
   snprintf(combinedname, totallen*sizeof(char), "%s(%s)",
            launchname, kernelname);
   vftr_veda_region_begin(combinedname);
   // Obtain stack and thread ID for the launched kernel
   // in order to pass it to the exit callback
   // to complete the kernel profiling
   thread_t *my_thread = vftr_get_my_thread(&(vftrace.process.threadtree));
   threadstack_t *my_threadstack = vftr_get_my_threadstack(my_thread);
   int stackID = my_threadstack->stackID;
   int threadID = my_thread->threadID;
   vftr_veda_region_end(combinedname);
   free(combinedname);

   // Store collect data that needs to be passed to
   // the launch_kernel_exit callback
   // This includes the starting timestamp
   // Stack and thread ID of the launched kernel
   user_data_t *user_data = (user_data_t*) malloc(sizeof(user_data_t));
   user_data->threadID = my_thread->threadID;
   user_data->stackID = my_threadstack->stackID;
   profile_t *my_prof = vftr_get_my_profile_from_ids(stackID, threadID);
   long long tend_callback = vftr_get_runtime_nsec();
   user_data->start_time = tend_callback;
   data->user_data = (void*) user_data;
   vftr_accumulate_veda_profiling_overhead(&(my_prof->vedaprof),
                                           tend_callback-tstart_callback);
}

void vftr_veda_callback_launch_kernel_exit(VEDAprofiler_data* data) {
   long long tstart_callback = vftr_get_runtime_nsec();
   long long kernel_end_time = tstart_callback;

   // get back user data
   user_data_t *user_data = (user_data_t*) data->user_data;
   long long runtime_usec = kernel_end_time - user_data->start_time;

   // store profiling data in profile
   profile_t *my_prof = vftr_get_my_profile_from_ids(user_data->stackID,
                                                   user_data->threadID);
   my_prof->vedaprof.total_time_nsec += runtime_usec;
   free(user_data);
   long long tend_callback = vftr_get_runtime_nsec();
   vftr_accumulate_veda_profiling_overhead(&(my_prof->vedaprof),
                                           tend_callback-tstart_callback);

}
