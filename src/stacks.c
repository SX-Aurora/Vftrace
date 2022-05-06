#include <stdlib.h>
#include <stdbool.h>

#include <string.h>

#include "stack_types.h"
#include "realloc_consts.h"
#include "stacks.h"
#include "profiling.h"
#include "symbols.h"
#include "hashing.h"
#include "search.h"
#include "collate_stacks.h"

void vftr_stacktree_realloc(stacktree_t *stacktree_ptr) {
   stacktree_t stacktree = *stacktree_ptr;
   while (stacktree.nstacks > stacktree.maxstacks) {
      int maxstacks = stacktree.maxstacks*vftr_realloc_rate+vftr_realloc_add;
      stacktree.stacks = (stack_t*)
         realloc(stacktree.stacks, maxstacks*sizeof(stack_t));
      stacktree.maxstacks = maxstacks;
   }
   *stacktree_ptr = stacktree;
}

void vftr_stack_callees_realloc(stack_t *stack_ptr) {
   stack_t stack = *stack_ptr;
   while (stack.ncallees > stack.maxcallees) {
      int maxcallees = stack.maxcallees*vftr_realloc_rate+vftr_realloc_add;
      stack.callees = (int*)
         realloc(stack.callees, maxcallees*sizeof(int));
      stack.maxcallees = maxcallees;
   }
   *stack_ptr = stack;
}

void vftr_insert_callee(int calleeID, stack_t *caller) {
   // append the callee to the list
   int idx = caller->ncallees;
   caller->ncallees++;
   vftr_stack_callees_realloc(caller);
   caller->callees[idx] = calleeID;
}

int vftr_new_stack(int callerID, stacktree_t *stacktree_ptr,
                   const char *name, stack_kind_t stack_kind,
                   uintptr_t address, bool precise) {
   int stackID = stacktree_ptr->nstacks;
   stacktree_ptr->nstacks++;
   vftr_stacktree_realloc(stacktree_ptr);

   stack_t *callerstack = stacktree_ptr->stacks+callerID;

   stack_t *stack = stacktree_ptr->stacks+stackID;
   stack->stack_kind = stack_kind;
   stack->address = address;
   stack->precise = precise;
   stack->caller = callerstack->lid;
   stack->maxcallees = 0;
   stack->ncallees = 0;
   stack->callees = NULL;
   stack->lid = stackID;

   stack->name = strdup(name);

   stack->profiling = vftr_new_profiling();
   vftr_insert_callee(stack->lid, callerstack);

   return stack->lid;
}

stack_t vftr_first_stack() {
   stack_t stack;
   stack.stack_kind = init;
   stack.address = (uintptr_t) NULL;
   stack.precise = false;
   stack.caller = -1;
   stack.maxcallees = 0;
   stack.ncallees = 0;
   stack.callees = NULL;
   stack.lid = 0;
   stack.name = strdup("init");
   stack.profiling = vftr_new_profiling();
   return stack;
}

// recursively free the stack tree
void vftr_stack_free(stack_t *stacks_ptr, int stackID) {
   stack_t stack = stacks_ptr[stackID];
   if (stack.ncallees > 0) {
      for (int icallee=0; icallee<stack.ncallees; icallee++) {
         vftr_stack_free(stacks_ptr, stack.callees[icallee]);
      }
      free(stack.callees);
      stack.callees = NULL;
      vftr_profiling_free(&(stack.profiling));
   } 
   free(stack.name);
   stacks_ptr[stackID] = stack;
}

stacktree_t vftr_new_stacktree() {
   stacktree_t stacktree;
   stacktree.maxstacks = 0;
   stacktree.stacks = NULL;
   stacktree.nstacks = 1;
   vftr_stacktree_realloc(&stacktree);
   stacktree.stacks[0] = vftr_first_stack();
   return stacktree;
}

void vftr_stacktree_free(stacktree_t *stacktree_ptr) {
   stacktree_t stacktree = *stacktree_ptr;
   if (stacktree.nstacks > 0) {
      vftr_stack_free(stacktree.stacks, 0);
      free(stacktree.stacks);
      stacktree.stacks = NULL;
      stacktree.nstacks = 0;
      stacktree.maxstacks = 0;
   }
   *stacktree_ptr = stacktree;
}

// fill in data that was not computed during runtime
void vftr_finalize_stacktree(stacktree_t *stacktree_ptr) {
   // exclusive time
   vftr_update_stacks_exclusive_time(stacktree_ptr->nstacks,
                                     stacktree_ptr->stacks);
}

void vftr_print_stack_branch(FILE *fp, int level, stacktree_t stacktree, int stackid) {
   // first print the indentation
   for (int ilevel=0; ilevel<level; ilevel++) {
      fprintf(fp, "  ");
   }
   stack_t stack = stacktree.stacks[stackid];
   fprintf(fp, "%s (%llx): id=%d, calls: %lld, incl_time/s: %lf\n",
           stack.name,
           (unsigned long long) stack.address,
           stack.lid,
           stack.profiling.callProf.calls,
           stack.profiling.callProf.time_usec*1.0e-6);

    
   level++;
   for (int icallee=0; icallee<stack.ncallees; icallee++) {
      vftr_print_stack_branch(fp, level, stacktree, stack.callees[icallee]);
   }
}

void vftr_print_stacktree(FILE *fp, stacktree_t stacktree) {
   vftr_print_stack_branch(fp, 0, stacktree, 0);
}

char *vftr_get_stack_string(stacktree_t stacktree, int stackid) {
   int stringlen = 0;
   int tmpstackid = stackid;
   stringlen += strlen(stacktree.stacks[stackid].name);
   stringlen ++; // function seperating character "<", or null terminator
   while (stacktree.stacks[tmpstackid].caller >= 0) {
      tmpstackid = stacktree.stacks[tmpstackid].caller;
      stringlen += strlen(stacktree.stacks[tmpstackid].name);
      stringlen ++; // function seperating character "<", or null terminator
//      if (stacktree.stacks[tmpstackid].precise) {
//         stringlen ++; // '*' for indicating precise functions
//      }
   }
   char *stackstring = (char*) malloc(stringlen*sizeof(char));
   // copy the chars one by one so there is no need to call strlen again.
   // thus minimizing reading the same memory locations over and over again.
   tmpstackid = stackid;
   char *tmpname_ptr = stacktree.stacks[tmpstackid].name;
   char *tmpstackstring_ptr = stackstring;
   while (*tmpname_ptr != '\0') {
      *tmpstackstring_ptr = *tmpname_ptr;
      tmpstackstring_ptr++;
      tmpname_ptr++;
   }
//   if (stacktree.stacks[tmpstackid].precise) {
//      *tmpstackstring_ptr = '*';
//      tmpstackstring_ptr++;
//   }
   while (stacktree.stacks[tmpstackid].caller >= 0) {
      // add function name separating character
      *tmpstackstring_ptr = '<';
      tmpstackstring_ptr++;
      tmpstackid = stacktree.stacks[tmpstackid].caller;
      char *tmpname_ptr = stacktree.stacks[tmpstackid].name;
      while (*tmpname_ptr != '\0') {
         *tmpstackstring_ptr = *tmpname_ptr;
         tmpstackstring_ptr++;
         tmpname_ptr++;
      }
//      if (stacktree.stacks[tmpstackid].precise) {
//         *tmpstackstring_ptr = '*';
//         tmpstackstring_ptr++;
//      }
   }
   // replace last char with a null terminator
   *tmpstackstring_ptr = '\0';
   return stackstring;
}

void vftr_print_stack(FILE *fp, stacktree_t stacktree, int stackid) {
   char *stackstr = vftr_get_stack_string(stacktree, stackid);
   fprintf(fp, "%s", stackstr);
   free(stackstr);
}

void vftr_print_stacklist(FILE *fp, stacktree_t stacktree) {
   for (int istack=0; istack<stacktree.nstacks; istack++) {
      fprintf(fp, "%d: ", istack);
      vftr_print_stack(fp, stacktree, istack);
      fprintf(fp, "\n");
   }
}
