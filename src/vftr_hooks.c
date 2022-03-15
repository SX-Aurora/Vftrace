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
#include "search.h"

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

   // check for recursion
   if (my_stack->address == func_addr) {
      // if recusive call, simply increas recursion depth count.
      // TODO: update performance and call counters as soon as implemented
      my_threadstack->recursion_depth++;
   } else {
      // search for the function in the stacks callees
      int calleeID = vftr_linear_search_callee(vftrace.process.stacktree.stacks,
                                               my_threadstack->stackID,
                                               func_addr);
      if (calleeID < 0) {
         // if the function was not found, create a new stack entry
         // and add its id to the callee list
         calleeID = vftr_new_stack(my_threadstack->stackID,
                                   &(vftrace.process.stacktree),
                                   vftrace.symboltable,
                                   function, (uintptr_t) func,
                                   true);
      }
      // push the function onto the threads stacklist
      vftr_threadstack_push(calleeID, &(my_thread->stacklist));
      // update the threadstack pointer
      my_threadstack = vftr_get_my_threadstack(my_thread);

      // accumulate profiling data
      // TODO: put in external function
      my_threadstack->profiling.callProf.time_usec -= function_entry_time_begin;
   }
   // increase callcount (this needs to be done regardless
   // whether it is recursive or not
   my_threadstack->profiling.callProf.calls++;

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
      threadstack_t function = vftr_threadstack_pop(&(my_thread->stacklist));

      stack_t *my_stack = vftrace.process.stacktree.stacks+function.stackID;
      vftr_accumulate_profiling(my_thread->master,
                                &(my_stack->profiling),
                                &(function.profiling));
      // No calls after this overhead handling
      // TODO: OMP distinquish between master and other threads
      my_stack->profiling.callProf.overhead_time_usec -= function_exit_time_begin
                                                         - vftr_get_runtime_usec();
   }
}
