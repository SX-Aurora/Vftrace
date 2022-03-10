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

void vftr_function_hook_entry(void *func, void *call_site) {
   // log function entry and exit time to estimate the overhead time
   //long long function_entry_time = vftr_get_runtime_usec();

   //thread_t *my_thread = vftr_get_my_thread(vftrace.process.threadtree);
  // stack_t *current_stack = vftrace.process.stacktree.stacks+my_thread->current_stackID;

  // int calleeID = vftr_linear_search_callee(vftrace.process.stacktree.stacks,
  //                                          my_thread->current_stackID,
  //                                          (uintptr_t) func);
  // //printf("Thread %d: %s calls", my_thread->thread_num, my_thread->current_stack->name);
  // if (calleeID < 0) {
  //    my_thread->current_stackID =  vftr_new_stack(my_thread->current_stackID,
  //                                                 &(vftrace.process.stacktree),
  //                                                 vftrace.symboltable,
  //                                                 function, (uintptr_t) func,
  //                                                 true);
  // } else {
  //    my_thread->current_stackID = calleeID;
  // }
  // //printf(" %s\n", my_thread->current_stack->name);
}

void vftr_function_hook_exit(void *func, void *call_site) {
   //thread_t *my_thread = vftr_get_my_thread(vftrace.process.threadtree);
   //stack_t *current_stack = vftrace.process.stacktree.stacks+my_thread->current_stackID;
   //printf("Thread %d: Leaving %s", my_thread->thread_num, my_thread->current_stack->name);
   //my_thread->current_stackID = current_stack->caller;
   //printf(" for %s\n", my_thread->current_stack->name);
}
