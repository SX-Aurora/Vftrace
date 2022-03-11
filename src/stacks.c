#include <stdlib.h>
#include <stdbool.h>
#ifdef _DEBUG
#include <stdio.h>
#endif

#include "stacks.h"
#include "symbols.h"
#include "search.h"

void vftr_stacktree_realloc(stacktree_t *stacktree_ptr) {
   stacktree_t stacktree = *stacktree_ptr;
   while (stacktree.nstacks > stacktree.maxstacks) {
      int maxstacks = stacktree.maxstacks*1.4+2;
      stacktree.stacks = (stack_t*)
         realloc(stacktree.stacks, maxstacks*sizeof(stack_t));
      stacktree.maxstacks = maxstacks;
   }
   *stacktree_ptr = stacktree;
}

void vftr_stack_callees_realloc(stack_t *stack_ptr) {
   stack_t stack = *stack_ptr;
   while (stack.ncallees > stack.maxcallees) {
      int maxcallees = stack.maxcallees*1.4+2;
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
   stack->gid = -1;

   // search for the function in the symbol table
   int symbID = vftr_binary_search_symboltable(symboltable.nsymbols,
                                               symboltable.symbols,
                                               address);
   if (symbID >= 0) {
      stack->name = symboltable.symbols[symbID].name;
   } else {
      stack->name = "(UnknownFunctionName)";
   }

   //printf("Caller: %s, ", caller->name);
   //printf("Callee: %s\n", stack->name);
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
   stack.gid = -1;
   stack.name = "init";
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

#ifdef _DEBUG
void vftr_print_stack(FILE *fp, int level, stacktree_t stacktree, int stackid) {
   // first print the indentation
   for (int ilevel=0; ilevel<level; ilevel++) {
      fprintf(fp, "  ");
   }
   stack_t stack = stacktree.stacks[stackid];
   fprintf(fp, "%s (%llx): id=%d\n", stack.name,
           (unsigned long long) stack.address, stack.lid);
   level++;
   for (int icallee=0; icallee<stack.ncallees; icallee++) {
      vftr_print_stack(fp, level, stacktree, stack.callees[icallee]);
   }
}

void vftr_print_stacktree(FILE *fp, stacktree_t stacktree) {
   fprintf(fp, "Stacktree\n");
   vftr_print_stack(fp, 0, stacktree, 0);
}
#endif
