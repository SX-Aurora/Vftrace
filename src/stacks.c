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

void vftr_insert_callee(int *ncallees_ptr, stack_t *callee, stack_t ***callees_ptr) {
   stack_t **callees = *callees_ptr;
   int ncallees = *ncallees_ptr;
   ncallees++;
   if (ncallees == 1) {
      callees = (stack_t**) malloc(sizeof(stack_t*));
      callees[0] = callee;
   } else {
      stack_t **tmp_callees;
      tmp_callees = (stack_t**) malloc(sizeof(stack_t*));
      int insertidx = ncallees-1;
      for (int idx=0; idx<ncallees-1; idx++) {
         if (callee->address < callees[idx]->address) {
            insertidx = idx;
            break;
         }
         tmp_callees[idx] = callees[idx];
      }
      tmp_callees[insertidx] = callee;
      for (int idx=insertidx; idx<ncallees-1; idx++) {
         tmp_callees[idx+1] = callees[idx];
      }

      free(callees);
      callees = tmp_callees;
   }

   *ncallees_ptr = ncallees;
   *callees_ptr = callees;
}

stack_t *vftr_new_stack(stack_t *caller, stacktree_t *stacktree_ptr,
                        symboltable_t symboltable, stack_kind_t stack_kind,
                        uintptr_t address, bool precise) {
   int stackID = stacktree_ptr->nstacks;
   stacktree_ptr->nstacks++;
   vftr_stacktree_realloc(stacktree_ptr);

   stack_t *stack = stacktree_ptr->stacks+stackID;
   stack->stack_kind = stack_kind;
   stack->address = address;
   stack->precise = precise;
   stack->caller = caller;
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

   vftr_insert_callee(&(caller->ncallees),
                      stack,
                      &(caller->callees));

   return stack;
}

stack_t vftr_first_stack() {
   stack_t stack;
   stack.stack_kind = init;
   stack.address = (uintptr_t) NULL;
   stack.precise = true;
   stack.caller = NULL;
   stack.ncallees = 0;
   stack.callees = NULL;
   stack.lid = 0;
   stack.gid = -1;
   stack.name = "init";
   return stack;
}

// recursively free the stack tree
void vftr_stack_free(stack_t *stack_ptr) {
   stack_t stack = *stack_ptr;
   if (stack.ncallees > 0) {
      for (int icallee=0; icallee<stack.ncallees; icallee++) {
         vftr_stack_free(stack.callees[icallee]);
      }
      free(stack.callees);
      stack.callees = NULL;
   } 
   *stack_ptr = stack;
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
      vftr_stack_free(stacktree.stacks+0);
      free(stacktree.stacks);
      stacktree.stacks = NULL;
      stacktree.nstacks = 0;
      stacktree.maxstacks = 0;
   }
   *stacktree_ptr = stacktree;
}

#ifdef _DEBUG
void vftr_print_stack(int level, stack_t stack) {
   // first print the indentation
   for (int ilevel=0; ilevel<level; ilevel++) {
      printf("  ");
   }
   printf("%s (%llx): id=%d\n", stack.name,
          (unsigned long long) stack.address, stack.lid);
   level++;
   for (int icallee=0; icallee<stack.ncallees; icallee++) {
      vftr_print_stack(level, *(stack.callees[icallee]));
   }
}

void vftr_print_stacktree(stacktree_t stacktree) {
   vftr_print_stack(0, stacktree.stacks[0]);
}
#endif
