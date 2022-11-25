#include <stdlib.h>

#include <veda.h>

#include "stack_types.h"
#include "thread_types.h"
#include "threadstack_types.h"
#include "vftrace_state.h"
#include "stacks.h"
#include "threads.h"
#include "threadstacks.h"
#include "profiling.h"

#include "timer.h"
#include "veda_regions.h"

typedef struct {
   long long start_time;
   int threadID;
   int stackID;
} user_data_t;

void vftr_veda_callback_launch_kernel_enter(VEDAprofiler_data* data) {
   // Start regions in order to generate
   // stack entries
   vftr_veda_region_begin("vedaLaunchKernel");
   VEDAprofiler_vedaLaunchKernel *LaunchKernelData;
   LaunchKernelData = (VEDAprofiler_vedaLaunchKernel*) &(data->type);
   vftr_veda_region_begin(LaunchKernelData->kernel);

   // Obtain stack and thread ID for the launched kernel
   // in order to pass it to the exit callback
   // to complete the kernel profiling
   thread_t *my_thread = vftr_get_my_thread(&(vftrace.process.threadtree));
   threadstack_t *my_threadstack = vftr_get_my_threadstack(my_thread);

   vftr_veda_region_end(LaunchKernelData->kernel);

   // Store collect data that needs to be passed to
   // the launch_kernel_exit callback
   // This includes the starting timestamp
   // Stack and thread ID of the launched kernel
   user_data_t *user_data = (user_data_t*) malloc(sizeof(user_data_t));
   user_data->threadID = thread_t->threadID;
   user_data->stackID = my_threadstack->stackID;
   user_data->start_time = vftr_get_runtime_nsec();
   data->user_data = (void*) user_data;
   vftr_veda_region_end("vedaLaunchKernel");
}

void vftr_veda_callback_launch_kernel_exit(VEDAprofiler_data* data) {
   long long kernel_end_time = vftr_get_runtime_nsec();
   // get back user data
   user_data_t *user_data = (user_data_t*) data->user_data;
   long long runtime_usec = kernel_end_time - user_data->start_time;
   printf("Kerneltime = %lld\n", runtime_usec);

   free(user_data);
}
