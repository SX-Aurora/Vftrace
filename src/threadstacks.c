#include <stdlib.h>
#include <stdio.h>

#include <string.h>
#include "realloc_consts.h"
#include "thread_types.h"
#include "threadstack_types.h"

#include "self_profile.h"
#include "threadstacks.h"
#include "search.h"

void vftr_threadstacklist_realloc(threadstacklist_t *stacklist_ptr) {
   SELF_PROFILE_START_FUNCTION;
   threadstacklist_t stacklist = *stacklist_ptr;
   while (stacklist.nstacks > stacklist.maxstacks) {
      int maxstacks = stacklist.maxstacks*vftr_realloc_rate+vftr_realloc_add;
      stacklist.stacks = (threadstack_t*)
         realloc(stacklist.stacks, maxstacks*sizeof(threadstack_t));
      stacklist.maxstacks = maxstacks;
   }
   *stacklist_ptr = stacklist;
   SELF_PROFILE_END_FUNCTION;
}

threadstacklist_t vftr_new_threadstacklist(int stackID) {
   SELF_PROFILE_START_FUNCTION;
   threadstacklist_t stacklist;
   stacklist.nstacks = 0;
   stacklist.maxstacks = 0;
   stacklist.stacks = NULL;
   if (stackID >= 0) {
      vftr_threadstack_push(stackID, &stacklist);
   }
   SELF_PROFILE_END_FUNCTION;
   return stacklist;
}

void vftr_threadstack_free(threadstack_t *stack_ptr) {
   threadstack_t stack = *stack_ptr;
   *stack_ptr = stack;
}

// push a new callstack onto the threads local stack
void vftr_threadstack_push(int stackID, threadstacklist_t *stacklist_ptr) {
   SELF_PROFILE_START_FUNCTION;
   threadstack_t stack;
   stack.stackID = stackID;
   stack.recursion_depth = 0;
   int idx = stacklist_ptr->nstacks;
   stacklist_ptr->nstacks++;
   vftr_threadstacklist_realloc(stacklist_ptr);
   stacklist_ptr->stacks[idx] = stack;
   SELF_PROFILE_END_FUNCTION;
}

threadstack_t *vftr_threadstack_pop(threadstacklist_t *stacklist_ptr) {
   SELF_PROFILE_START_FUNCTION;
   stacklist_ptr->nstacks--;
   threadstack_t *threadstack = stacklist_ptr->stacks + stacklist_ptr->nstacks-1;
   SELF_PROFILE_END_FUNCTION;
   return threadstack;
}

void vftr_threadstacklist_free(threadstacklist_t *stacklist_ptr) {
   SELF_PROFILE_START_FUNCTION;
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
   SELF_PROFILE_END_FUNCTION;
}

threadstack_t *vftr_get_my_threadstack(thread_t *my_thread_ptr) {
   SELF_PROFILE_START_FUNCTION;
   int idx = my_thread_ptr->stacklist.nstacks;
   threadstack_t *mythreadstack = NULL;
   if (idx != 0) {
      mythreadstack = my_thread_ptr->stacklist.stacks + idx - 1;
   }
   SELF_PROFILE_END_FUNCTION;
   return mythreadstack;
}

threadstack_t *vftr_update_threadstack_function(threadstack_t *my_threadstack,
                                                thread_t *my_thread,
                                                uintptr_t func_addr,
                                                vftrace_t *vftrace) {
   SELF_PROFILE_START_FUNCTION;
   // search for the function in the stacks callees
   int calleeID = vftr_linear_search_callee(vftrace->process.stacktree.stacks,
                                            my_threadstack->stackID,
                                            func_addr);
   if (calleeID < 0) {
      int symbID = vftr_get_symbID_from_address(vftrace->symboltable, func_addr);
      char *name = vftr_get_name_from_symbID(vftrace->symboltable, symbID);
      char *cleanname = vftr_get_cleanname_from_symbID(vftrace->symboltable, symbID);
      bool precise = vftr_get_preciseness_from_symbID(vftrace->symboltable, symbID);
      // if the function was not found, create a new stack entry
      // and add its id to the callee list
      calleeID = vftr_new_stack(my_threadstack->stackID,
                                &(vftrace->process.stacktree),
                                name, cleanname, func_addr,
                                precise);
   }
   // push the function onto the threads stacklist
   vftr_threadstack_push(calleeID, &(my_thread->stacklist));
   // update the threadstack pointer
   threadstack_t *new_threadstack = vftr_get_my_threadstack(my_thread);
   SELF_PROFILE_END_FUNCTION;
   return new_threadstack;
}

threadstack_t *vftr_update_threadstack_region(threadstack_t *my_threadstack,
                                              thread_t *my_thread,
                                              uintptr_t region_addr,
                                              const char *name,
                                              vftrace_t *vftrace,
                                              bool precise) {
   SELF_PROFILE_START_FUNCTION;
   // search for the function in the stacks callees
   int calleeID = vftr_linear_search_callee(vftrace->process.stacktree.stacks,
                                            my_threadstack->stackID,
                                            region_addr);
   if (calleeID < 0 ||
       strcmp(vftrace->process.stacktree.stacks[calleeID].name, name)) {
      // if the function was not found, create a new stack entry
      // and add its id to the callee list
      calleeID = vftr_new_stack(my_threadstack->stackID,
                                &(vftrace->process.stacktree),
                                name, name,
                                region_addr,
                                precise);
   }
   // push the function onto the threads stacklist
   vftr_threadstack_push(calleeID, &(my_thread->stacklist));
   // update the threadstack pointer
   threadstack_t *new_threadstack = vftr_get_my_threadstack(my_thread);
   SELF_PROFILE_END_FUNCTION;
   return new_threadstack;
}

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
