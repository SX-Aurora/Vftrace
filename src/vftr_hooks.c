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

#include "stack_types.h"
#include "thread_types.h"
#include "threadstack_types.h"
#include "vftrace_state.h"

#include "stacks.h"
#include "threads.h"
#include "threadstacks.h"
#include "profiling.h"
#include "timer.h"

void vftr_function_entry(void *func, void *call_site) {
   long long function_entry_time_begin = vftr_get_runtime_usec();

   // Get the thread that called the function
   thread_t *my_thread = vftr_get_my_thread(&(vftrace.process.threadtree));
   threadstack_t *my_threadstack = vftr_get_my_threadstack(my_thread);
   // TODO: when OMP support is implemented it must be checked
   //       whether to inherit the parentthreads stack + the function, or 
   //       to inherit it as soon as a task is created. for non-OMP code the master
   //       thread is created with _init as lowest stacklist entry

   vftr_print_threadtree(stderr, vftrace.process.threadtree);
   vftr_print_stacktree(stderr, vftrace.process.stacktree);
   stack_t *my_stack = vftrace.process.stacktree.stacks+my_threadstack->stackID;
   
   // cast and store function address once, as it is needed multiple times
   uintptr_t func_addr = (uintptr_t) func;

   // TODO: update performance and call counters as soon as implemented
   // check for recursion
   if (my_stack->address == func_addr) {
      // if recusive call, simply increas recursion depth count.
      my_threadstack->recursion_depth++;
      my_threadstack->profiling.callProf.calls++;
   } else {
      // add possibly new functions to the stack
      // and adjust the threadstack accordingly
      my_threadstack = vftr_update_threadstack(my_threadstack, my_thread,
                                               func_addr, &vftrace);



      // accumulate profiling data
      // TODO: put in external function
      my_threadstack->profiling.callProf.time_usec -= function_entry_time_begin;
      my_threadstack->profiling.callProf.calls++;
   }

   // No calls after this overhead handling!
   my_threadstack->profiling.callProf.overhead_time_usec -= function_entry_time_begin
                                                            - vftr_get_runtime_usec();
}

void vftr_function_exit(void *func, void *call_site) {
   long long function_exit_time_begin = vftr_get_runtime_usec();

   thread_t *my_thread = vftr_get_my_thread(&(vftrace.process.threadtree));
   threadstack_t *my_threadstack = vftr_get_my_threadstack(my_thread);

   // check if still in a recursive call
   if (my_threadstack->recursion_depth > 0) {
      // simply decrement the recursion depth counter
      my_threadstack->recursion_depth--;

      // No calls after this overhead handling!
      my_threadstack->profiling.callProf.overhead_time_usec -= function_exit_time_begin
                                                               - vftr_get_runtime_usec();
   } else {
      // accumulate threadded profilig data
      // TODO: put in external function
      my_threadstack->profiling.callProf.time_usec += function_exit_time_begin;

      // if not recursive pop the function from the threads stacklist
      my_threadstack = vftr_threadstack_pop(&(my_thread->stacklist));

      stack_t *my_stack = vftrace.process.stacktree.stacks+my_threadstack->stackID;
      vftr_accumulate_profiling(my_thread->master,
                                &(my_stack->profiling),
                                &(my_threadstack->profiling));


      // No calls after this overhead handling
      // TODO: OMP distinquish between master and other threads
      my_stack->profiling.callProf.overhead_time_usec -= function_exit_time_begin
                                                         - vftr_get_runtime_usec();
   }
}
