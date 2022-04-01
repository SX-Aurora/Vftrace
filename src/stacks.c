#include <stdlib.h>
#include <stdbool.h>

#include "realloc_consts.h"
#include "stacks.h"
#include "profiling.h"
#include "symbols.h"
#include "hashing.h"
#include "search.h"

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
                   symboltable_t symboltable, stack_kind_t stack_kind,
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

   // search for the function in the symbol table
   int symbID = vftr_binary_search_symboltable(symboltable.nsymbols,
                                               symboltable.symbols,
                                               address);
   if (symbID >= 0) {
      stack->name = symboltable.symbols[symbID].name;
   } else {
      stack->name = "(UnknownFunctionName)";
   }

   stack->profiling = vftr_new_profiling();
   vftr_insert_callee(stack->lid, callerstack);

   return stack->lid;
}

stack_t vftr_first_stack() {
   stack_t stack;
   stack.stack_kind = init;
   stack.address = (uintptr_t) NULL;
   stack.precise = true;
   stack.caller = -1;
   stack.maxcallees = 0;
   stack.ncallees = 0;
   stack.callees = NULL;
   stack.lid = 0;
   stack.name = "init";
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

   // compute stack hashes for normalization
   vftr_compute_stack_hashes(stacktree_ptr->nstacks,
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

void vftr_print_stack(FILE *fp, stacktree_t stacktree, int stackid) {
   fprintf(fp, "%s", stacktree.stacks[stackid].name);
   if (stacktree.stacks[stackid].caller >= 0) {
      fprintf(fp, "<");
      vftr_print_stack(fp, stacktree, stacktree.stacks[stackid].caller);
   }
}

void vftr_print_stacklist(FILE *fp, stacktree_t stacktree) {
   for (int istack=0; istack<stacktree.nstacks; istack++) {
      fprintf(fp, "%u: ", istack);
      vftr_print_stack(fp, stacktree, istack);
      fprintf(fp, "\n");
   }
}
