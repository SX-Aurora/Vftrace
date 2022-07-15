#include "thread_types.h"
#include "stack_types.h"

#include "vftrace_state.h"
#include "threads.h"
#include "threadstacks.h"
#include "stacks.h"

int vftrace_get_stack_string_length() {
   // Get the thread that called the function
   thread_t *my_thread = vftr_get_my_thread(&(vftrace.process.threadtree));
   threadstack_t *my_threadstack = vftr_get_my_threadstack(my_thread);
   // TODO: when OMP support is implemented it must be checked
   //       whether to inherit the parentthreads stack + the function, or
   //       to inherit it as soon as a task is created. for non-OMP code the master
   //       thread is created with _init as lowest stacklist entry
   return vftr_get_stack_string_length(vftrace.process.stacktree,
                                       my_threadstack->stackID, false);
}

char *vftrace_get_stack() {
   // Get the thread that called the function
   thread_t *my_thread = vftr_get_my_thread(&(vftrace.process.threadtree));
   threadstack_t *my_threadstack = vftr_get_my_threadstack(my_thread);
   // TODO: when OMP support is implemented it must be checked
   //       whether to inherit the parentthreads stack + the function, or
   //       to inherit it as soon as a task is created. for non-OMP code the master
   //       thread is created with _init as lowest stacklist entry
   return vftr_get_stack_string(vftrace.process.stacktree,
                                my_threadstack->stackID, false);
}
