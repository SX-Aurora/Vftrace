#include <stdlib.h>
#include <stdbool.h>

#include <string.h>

#include "self_profile.h"
#include "stack_types.h"
#include "thread_types.h"
#include "threadstack_types.h"
#include "vftrace_state.h"

#include "stacks.h"
#include "threads.h"
#include "threadstacks.h"
#include "profiling.h"
#include "callprofiling.h"
#include "sampling.h"
#include "timer.h"
#include "misc_utils.h"
#ifdef _OMP
#include <omp.h>
#endif

void vftr_omp_region_begin(const char *name, void *addr) {
   SELF_PROFILE_START_FUNCTION;
   long long region_begin_time_begin = vftr_get_runtime_nsec();
#ifdef _OMP
   omp_set_lock(&(vftrace.process.threadlock));
#endif
   // Get the thread that called the region
   thread_t *my_thread = vftr_get_my_thread(&(vftrace.process.threadtree));
   threadstack_t *my_threadstack = vftr_get_my_threadstack(my_thread);
   // TODO: when OMP support is implemented it must be checked
   //       whether to inherit the parentthreads stack + the region, or
   //       to inherit it as soon as a task is created. for non-OMP code the master
   //       thread is created with _init as lowest stacklist entry
   vftr_stack_t *my_stack = vftrace.process.stacktree.stacks+my_threadstack->stackID;
   profile_t *my_profile = vftr_get_my_profile(my_stack, my_thread);

   // cast and store region address once, as it is needed multiple times
   uintptr_t region_addr = (uintptr_t) addr;
   // TODO: update performance and call counters as soon as implemented
   // check for recursion
   // need to check for same address and name.
   // if a dynamically created region is called recuresively
   // it might have the same address, but the name can differ
   if (my_stack->address == region_addr && !strcmp(name, my_stack->name)) {
      // if recusive call, simply increas recursion depth count.
      my_threadstack->recursion_depth++;
      vftr_accumulate_callprofiling(&(my_profile->callprof), 1, 0);
   } else {
      // add possibly new region to the stack
      // and adjust the threadstack accordingly
      my_threadstack = vftr_update_threadstack_region(my_threadstack, my_thread,
                                                      region_addr, name,
                                                      &vftrace, true);
      // OMP regions move the thread level up by one
      // Therefore the next subthread level needs to be clean
      vftr_thread_subthreads_reset(&(vftrace.process.threadtree), my_thread->threadID);

      vftr_stack_t *my_new_stack = vftrace.process.stacktree.stacks+my_threadstack->stackID;
      my_profile = vftr_get_my_profile(my_new_stack, my_thread);

      vftr_sample_function_entry(&(vftrace.sampling),
                                 *my_new_stack,
                                 region_begin_time_begin);

      // accumulate profiling data
      vftr_accumulate_callprofiling(&(my_profile->callprof),
                                    1, -region_begin_time_begin);
   }

   // No calls after this overhead handling!
   vftr_accumulate_callprofiling_overhead(&(my_profile->callprof),
      vftr_get_runtime_nsec() - region_begin_time_begin);
#ifdef _OMP
   omp_unset_lock(&(vftrace.process.threadlock));
#endif
   SELF_PROFILE_END_FUNCTION;
}

void vftr_omp_region_end(const char *name) {
   SELF_PROFILE_START_FUNCTION;
   (void) name;
   long long region_end_time_begin = vftr_get_runtime_nsec();
#ifdef _OMP
   omp_set_lock(&(vftrace.process.threadlock));
#endif

   thread_t *my_thread = vftr_get_my_thread(&(vftrace.process.threadtree));
   threadstack_t *my_threadstack = vftr_get_my_threadstack(my_thread);
   vftr_stack_t *my_stack = vftrace.process.stacktree.stacks+my_threadstack->stackID;
   profile_t *my_profile = vftr_get_my_profile(my_stack, my_thread);

   // check if still in a recursive call
   if (my_threadstack->recursion_depth > 0) {
      // simply decrement the recursion depth counter
      my_threadstack->recursion_depth--;
   } else {
      // accumulate threadded profiling data
      vftr_accumulate_callprofiling(&(my_profile->callprof),
                                    0, region_end_time_begin);

      // if not recursive pop the function from the threads stacklist
      my_threadstack = vftr_threadstack_pop(&(my_thread->stacklist));

      vftr_stack_t *my_new_stack = vftrace.process.stacktree.stacks+my_threadstack->stackID;
      profile_t *my_new_profile = vftr_get_my_profile(my_new_stack, my_thread);

      // TODO Add accumulation of profiling data
      vftr_sample_function_exit(&(vftrace.sampling),
                                *my_stack,
                                region_end_time_begin);
   }

   // No calls after this overhead handling
   vftr_accumulate_callprofiling_overhead(
      &(my_profile->callprof),
      vftr_get_runtime_nsec() - region_end_time_begin);
#ifdef _OMP
   omp_unset_lock(&(vftrace.process.threadlock));
#endif
   SELF_PROFILE_END_FUNCTION;
}
