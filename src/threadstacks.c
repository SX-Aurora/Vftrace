#include <stdlib.h>
#ifdef _DEBUG
#include <stdio.h>
#endif

#include "realloc_consts.h"
#include "thread_types.h"
#include "threadstack_types.h"
#include "profiling.h"
#include "threadstacks.h"

void vftr_threadstacklist_realloc(threadstacklist_t *stacklist_ptr) {
   threadstacklist_t stacklist = *stacklist_ptr;
   while (stacklist.nstacks > stacklist.maxstacks) {
      int maxstacks = stacklist.maxstacks*vftr_realloc_rate+vftr_realloc_add;
      stacklist.stacks = (threadstack_t*)
         realloc(stacklist.stacks, maxstacks*sizeof(threadstack_t));
      stacklist.maxstacks = maxstacks;
   }
   *stacklist_ptr = stacklist;
}

threadstacklist_t vftr_new_threadstacklist(int stackID) {
   threadstacklist_t stacklist;
   stacklist.nstacks = 0;
   stacklist.maxstacks = 0;
   stacklist.stacks = NULL;
   if (stackID >= 0) {
      vftr_threadstack_push(stackID, &stacklist);
   }
   return stacklist;
}

void vftr_threadstack_free(threadstack_t *stack_ptr) {
   threadstack_t stack = *stack_ptr;
   vftr_profiling_free(&(stack.profiling));
   *stack_ptr = stack;
}

// push a new callstack onto the threads local stack
void vftr_threadstack_push(int stackID, threadstacklist_t *stacklist_ptr) {
   threadstack_t stack;
   stack.stackID = stackID;
   stack.recursion_depth = 0;
   // TODO: enable profiling
   stack.profiling = vftr_new_profiling();
   int idx = stacklist_ptr->nstacks;
   stacklist_ptr->nstacks++;
   vftr_threadstacklist_realloc(stacklist_ptr);
   stacklist_ptr->stacks[idx] = stack;
}

threadstack_t vftr_threadstack_pop(threadstacklist_t *stacklist_ptr) {
   stacklist_ptr->nstacks--;
   return stacklist_ptr->stacks[stacklist_ptr->nstacks];
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

threadstack_t *vftr_get_my_threadstack(thread_t *my_thread_ptr) {
   int idx = my_thread_ptr->stacklist.nstacks;
   if (idx == 0) {
      return NULL;
   } else {
      return my_thread_ptr->stacklist.stacks+idx-1;
   }
}

#ifdef _DEBUG
void vftr_print_threadstack(FILE *fp, threadstacklist_t stacklist) {
   if (stacklist.nstacks == 0) {
      fprintf(fp, "None");
   } else {
      fprintf(fp, "%d", stacklist.stacks[0].stackID);
      if (stacklist.stacks[0].recursion_depth > 0) {
         fprintf(fp, "(%d)", stacklist.stacks[0].recursion_depth);
      }
      for (int istack=1; istack<stacklist.nstacks; istack++) {
         fprintf(fp, "->%d", stacklist.stacks[istack].stackID);
         if (stacklist.stacks[istack].recursion_depth > 0) {
            fprintf(fp, "(%d)", stacklist.stacks[istack].recursion_depth);
         }
      }
   }
}
#endif
