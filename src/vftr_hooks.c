/*
   This file is part of Vftrace.

   Vftrace is free software; you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation; either version 2 of the License, or
   (at your option) any later version.

   Vftrace is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License along
   with this program; if not, write to the Free Software Foundation, Inc.,
   51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
*/
#include <stdio.h>

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
#ifdef _MPI
#include "requests.h"
#endif

void vftr_function_entry(void *func, void *call_site) {
   SELF_PROFILE_START_FUNCTION;
   (void) call_site;
   long long function_entry_time_begin = vftr_get_runtime_usec();

#ifdef _MPI
   // Check for completed MPI-requests
   vftr_clear_completed_requests_from_hooks();
#endif

   // Get the thread that called the function
   thread_t *my_thread = vftr_get_my_thread(&(vftrace.process.threadtree));
   threadstack_t *my_threadstack = vftr_get_my_threadstack(my_thread);
   // TODO: when OMP support is implemented it must be checked
   //       whether to inherit the parentthreads stack + the function, or
   //       to inherit it as soon as a task is created. for non-OMP code the master
   //       thread is created with _init as lowest stacklist entry
   stack_t *my_stack = vftrace.process.stacktree.stacks+my_threadstack->stackID;
   profile_t *my_profile = vftr_get_my_profile(my_stack, my_thread);

   // cast and store function address once, as it is needed multiple times
   uintptr_t func_addr = (uintptr_t) func;

   // check for recursion
   if (my_stack->address == func_addr) {
      // if recusive call, simply increas recursion depth count.
      my_threadstack->recursion_depth++;
      vftr_accumulate_callprofiling(&(my_profile->callProf), 1, 0);
   } else {
      // add possibly new functions to the stack
      // and adjust the threadstack accordingly
      my_threadstack = vftr_update_threadstack_function(my_threadstack, my_thread,
                                                        func_addr, &vftrace);
      stack_t *my_new_stack = vftrace.process.stacktree.stacks+my_threadstack->stackID;
      my_profile = vftr_get_my_profile(my_new_stack, my_thread);

      vftr_sample_function_entry(&(vftrace.sampling),
                                 *my_new_stack,
                                 function_entry_time_begin);


      // accumulate call profiling data
      vftr_accumulate_callprofiling(&(my_profile->callProf),
                                    1, -function_entry_time_begin);
   }

   // No calls after this overhead handling!
   vftr_accumulate_callprofiling_overhead(&(my_profile->callProf),
      vftr_get_runtime_usec() - function_entry_time_begin);
   SELF_PROFILE_END_FUNCTION;
}

void vftr_function_exit(void *func, void *call_site) {
   SELF_PROFILE_START_FUNCTION;
   (void) func;
   (void) call_site;
   long long function_exit_time_begin = vftr_get_runtime_usec();

#ifdef _MPI
   // Check for completed MPI-requests
   vftr_clear_completed_requests_from_hooks();
#endif

   thread_t *my_thread = vftr_get_my_thread(&(vftrace.process.threadtree));
   threadstack_t *my_threadstack = vftr_get_my_threadstack(my_thread);
   stack_t *my_stack = vftrace.process.stacktree.stacks+my_threadstack->stackID;
   profile_t *my_profile = vftr_get_my_profile(my_stack, my_thread);

   // check if still in a recursive call
   if (my_threadstack->recursion_depth > 0) {
      // simply decrement the recursion depth counter
      my_threadstack->recursion_depth--;
   } else {
      // accumulate threadded profiling data
      vftr_accumulate_callprofiling(&(my_profile->callProf),
                                    0, function_exit_time_begin);

      // if not recursive pop the function from the threads stacklist
      my_threadstack = vftr_threadstack_pop(&(my_thread->stacklist));

      stack_t *my_new_stack = vftrace.process.stacktree.stacks+my_threadstack->stackID;
      profile_t *my_new_profile = vftr_get_my_profile(my_new_stack, my_thread);

      // TODO Add accumulation of profiling data
      vftr_sample_function_exit(&(vftrace.sampling),
                                *my_new_stack,
                                function_exit_time_begin);
   }
   // No calls after this overhead handling
   vftr_accumulate_callprofiling_overhead(
      &(my_profile->callProf),
      vftr_get_runtime_usec() - function_exit_time_begin);
   SELF_PROFILE_END_FUNCTION;
}
