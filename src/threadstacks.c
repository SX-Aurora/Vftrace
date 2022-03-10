#include <stdlib.h>

#include "threadstacks.h"

void vftr_threadstacklist_realloc(threadstacklist_t *stacklist_ptr) {
   threadstacklist_t stacklist = *stacklist_ptr;
   while (stacklist.nstacks > stacklist.maxstacks) {
      int maxstacks = stacklist.maxstacks*1.4+2;
      stacklist.stacks = (threadstack_t*)
         realloc(stacklist.stacks, maxstacks*sizeof(threadstack_t));
      stacklist.maxstacks = maxstacks;
   }
   *stacklist_ptr = stacklist;
}

threadstacklist_t vftr_new_threadstacklist() {
   threadstacklist_t stacklist;
   stacklist.nstacks = 0;
   stacklist.maxstacks = 0;
   stacklist.stacks = NULL;
   return stacklist;
}

void vftr_threadstack_free(threadstack_t *stack_ptr) {
   threadstack_t stack = *stack_ptr;
   // TODO: free profiling data
   *stack_ptr = stack;
}

// push a new callstack onto the threads local stack
void vftr_threadstack_push(int stackID, threadstacklist_t *stacklist_ptr) {
   threadstack_t stack;
   stack.stackID = stackID;
   stack.recursion_depth = 0;
   // TODO: enable profiling
   // stack.profiling = vftr_new_profiling();
   int idx = stacklist_ptr->nstacks;
   stacklist_ptr->nstacks++;
   vftr_threadstacklist_realloc(stacklist_ptr);
   stacklist_ptr->stacks[idx] = stack;
}

threadstack_t vftr_threadstack_pop(threadstacklist_t *stacklist_ptr) {
   int idx = stacklist_ptr->nstacks;
   stacklist_ptr->nstacks--;
   return stacklist_ptr->stacks[idx];
}

void vftr_threadstacklist_free(threadstacklist_t *stacklist_ptr) {
   threadstacklist_t stacklist = *stacklist_ptr;
   if (stacklist.maxstacks > 0) {
      for (int istack=0; istack<stacklist.nstacks; istack++) {
         vftr_threadstack_free(stacklist.stacks+istack);
      }
      free(stacklist.stacks);
      stacklist.stacks = NULL;
      stacklist.nstacks = 0;
      stacklist.maxstacks = 0;
   }
   *stacklist_ptr = stacklist;
}


