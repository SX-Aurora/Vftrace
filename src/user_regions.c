#include <stdlib.h>

#include <string.h>

#include "stack_types.h"
#include "thread_types.h"
#include "threadstack_types.h"
#include "vftrace_state.h"

#include "stacks.h"
#include "threads.h"
#include "threadstacks.h"
#include "profiling.h"
#include "sampling.h"
#include "timer.h"

void vftr_user_region_begin(const char *name, void *addr) {
   long long region_begin_time_begin = vftr_get_runtime_usec();
   // Get the thread that called the region
   thread_t *my_thread = vftr_get_my_thread(&(vftrace.process.threadtree));
   threadstack_t *my_threadstack = vftr_get_my_threadstack(my_thread);
   // TODO: when OMP support is implemented it must be checked
   //       whether to inherit the parentthreads stack + the region, or
   //       to inherit it as soon as a task is created. for non-OMP code the master
   //       thread is created with _init as lowest stacklist entry
   stack_t *my_stack = vftrace.process.stacktree.stacks+my_threadstack->stackID;

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
      my_threadstack->profiling.callProf.calls++;
   } else {
      // add possibly new region to the stack
      // and adjust the threadstack accordingly
      my_threadstack = vftr_update_threadstack_region(my_threadstack, my_thread,
                                                      region_addr, name, &vftrace);
 
      vftr_sample_function_entry(&(vftrace.sampling),
                                 my_threadstack->stackID,
                                 region_begin_time_begin);
 
 
      // accumulate profiling data
      // TODO: put in external function
      my_threadstack->profiling.callProf.time_usec -= region_begin_time_begin;
      my_threadstack->profiling.callProf.calls++;
   }
 
   // No calls after this overhead handling!
   my_threadstack->profiling.callProf.overhead_time_usec -= region_begin_time_begin
                                                            - vftr_get_runtime_usec();
}

void vftr_user_region_end() {
   long long function_end_time_begin = vftr_get_runtime_usec();

   thread_t *my_thread = vftr_get_my_thread(&(vftrace.process.threadtree));
   threadstack_t *my_threadstack = vftr_get_my_threadstack(my_thread);

   // check if still in a recursive call
   if (my_threadstack->recursion_depth > 0) {
      // simply decrement the recursion depth counter
      my_threadstack->recursion_depth--;

      // No calls after this overhead handling!
      my_threadstack->profiling.callProf.overhead_time_usec -= function_end_time_begin
                                                               - vftr_get_runtime_usec();
   } else {
      // accumulate threadded profilig data
      // TODO: put in external function
      my_threadstack->profiling.callProf.time_usec += function_end_time_begin;

      // if not recursive pop the function from the threads stacklist
      my_threadstack = vftr_threadstack_pop(&(my_thread->stacklist));

      stack_t *my_stack = vftrace.process.stacktree.stacks+my_threadstack->stackID;
      vftr_accumulate_profiling(my_thread->master,
                                &(my_stack->profiling),
                                &(my_threadstack->profiling));

      vftr_sample_function_exit(&(vftrace.sampling), my_threadstack->stackID,
                                function_end_time_begin);

      // No calls after this overhead handling
      // TODO: OMP distinquish between master and other threads
      my_stack->profiling.callProf.overhead_time_usec -= function_end_time_begin
                                                         - vftr_get_runtime_usec();
   }
}

// Getting the region address is defined here as a macro,
// so it wont mess up the adresses by changing the function stack
#ifdef __ve__
#define GET_REGION_ADDRESS(ADDR) asm volatile ("or %0,0,%%lr" : "=r" (ADDR))
#else
#define GET_REGION_ADDRESS(ADDR) asm volatile ("mov 8(%%rbp), %0" : "=r" (ADDR))
#endif

//These regions are for users to be used only.
void vftrace_region_begin(const char *name) {
   if (vftrace.state == on) {
      void *addr;
      GET_REGION_ADDRESS(addr);
      vftr_user_region_begin(name, addr);
   }
}

void vftrace_region_end(const char *name) {
   (void) name;
   if (vftrace.state == on) {
      vftr_user_region_end();
   }
}


#undef GET_REGION_ADDRESS
