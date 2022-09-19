#include <stdlib.h>
#include <stdio.h>

#include <string.h>

#include "realloc_consts.h"
#include "function_types.h"
#include "stack_types.h"

int search_callee(stack_t *stacks, int callerID, char *name) {
   stack_t stack = stacks[callerID];
   int calleeID = -1;
   for (int icallee=0; icallee<stack.ncallees; icallee++) {
      int stackID = stack.callees[icallee];
      stack_t calleestack = stacks[stackID];
      if (strcmp(name, calleestack.name) == 0) {
         calleeID = stackID;
         break;
      }
   }

   return calleeID;
}

void stack_callees_realloc(stack_t *stack_ptr) {
   stack_t stack = *stack_ptr;
   while (stack.ncallees > stack.maxcallees) {
      int maxcallees = stack.maxcallees*realloc_rate+realloc_add;
      stack.callees = (int*)
         realloc(stack.callees, maxcallees*sizeof(int));
      stack.maxcallees = maxcallees;
   }
   *stack_ptr = stack;
}

void stacktree_realloc(stacktree_t *stacktree_ptr) {
   stacktree_t stacktree = *stacktree_ptr;
   while (stacktree.nstacks > stacktree.maxstacks) {
      int maxstacks = stacktree.maxstacks*realloc_rate+realloc_add;
      stacktree.stacks = (stack_t*)
         realloc(stacktree.stacks, maxstacks*sizeof(stack_t));
      stacktree.maxstacks = maxstacks;
   }
   *stacktree_ptr = stacktree;
}

void insert_callee(int calleeID, stack_t *caller) {
   int idx = caller->ncallees;
   caller->ncallees++;
   stack_callees_realloc(caller);
   caller->callees[idx] = calleeID;
}

int new_stack(int callerID, char *name, stacktree_t *stacktree_ptr) {
   int stackID = stacktree_ptr->nstacks;
   stacktree_ptr->nstacks++;
   stacktree_realloc(stacktree_ptr);

   stack_t *callerstack = stacktree_ptr->stacks+callerID;

   stack_t *stack = stacktree_ptr->stacks+stackID;
   stack->caller = callerstack->id;
   stack->maxcallees = 0;
   stack->ncallees = 0;
   stack->callees = NULL;
   stack->id = stackID;

   stack->name = name;
   stack->ncalls = 0;
   stack->time_nsec = 0;
   stack->time_excl_nsec = 0;

   insert_callee(stack->id, callerstack);

   return stack->id;
}

stack_t first_stack(char *name) {
   stack_t stack;
   stack.name = name;
   stack.id = 0;
   stack.caller = -1;
   stack.maxcallees = 0;
   stack.ncallees = 0;
   stack.callees = NULL;
   stack.ncalls = 0;
   stack.time_nsec = 0;
   stack.time_excl_nsec = 0;
   return stack;
}

void free_stack(stack_t *stacks_ptr, int stackID) {
   stack_t stack = stacks_ptr[stackID];
   if (stack.ncallees > 0) {
      for (int icallee=0; icallee<stack.ncallees; icallee++) {
         free_stack(stacks_ptr, stack.callees[icallee]);
      }
      free(stack.callees);
      stack.callees = NULL;
   }
   stacks_ptr[stackID] = stack;
}

stacktree_t new_stacktree(functionlist_t functionlist) {
   stacktree_t stacktree;
   stacktree.maxstacks = 0;
   stacktree.stacks = NULL;
   stacktree.nstacks = 1;
   stacktree_realloc(&stacktree);
   stacktree.stacks[0] = first_stack(functionlist.functions[0].name);
   return stacktree;
}

void free_stacktree(stacktree_t *stacktree_ptr) {
   if (stacktree_ptr->nstacks > 0) {
      free_stack(stacktree_ptr->stacks, 0);
      free(stacktree_ptr->stacks);
      stacktree_ptr->nstacks = 0;
      stacktree_ptr->maxstacks = 0;
   }
}

void update_stacks_exclusive_time(stacktree_t *stacktree_ptr) {
   int nstacks = stacktree_ptr->nstacks;
   stack_t *stacks = stacktree_ptr->stacks;
   // exclusive time for init is 0, therefore it does not need to be computed.
   for (int istack=1; istack<nstacks; istack++) {
      stack_t *mystack = stacks + istack;
      mystack->time_excl_nsec = mystack->time_nsec;
      for (int icallee=0; icallee<mystack->ncallees; icallee++) {
         int calleeID = mystack->callees[icallee];
         stack_t *calleestack = stacks+calleeID;
         mystack->time_excl_nsec -= calleestack->time_nsec;
      }
   }
}

void finalize_stacktree(stacktree_t *stacktree_ptr) {
   // exclusive time
   update_stacks_exclusive_time(stacktree_ptr);
}

void qsort_stacklist(int n, stack_t **stacklist) {
   if (n < 2) return;
   stack_t *pivot = stacklist[n/2];
   int left, right;
   for (left=0, right=n-1; ; left++, right--) {
      while (stacklist[left]->time_excl_nsec > pivot->time_excl_nsec) left++;
      while (stacklist[right]->time_excl_nsec < pivot->time_excl_nsec) right--;
      if (left >= right) break;
      stack_t *temp = stacklist[left];
      stacklist[left] = stacklist[right];
      stacklist[right] = temp;
   }
   qsort_stacklist(left, stacklist);
   qsort_stacklist(n-left, stacklist+left);
}

stack_t **sort_stacks_by_excl_time(stacktree_t stacktree) {
   int nstacks = stacktree.nstacks;
   stack_t **stacklist = (stack_t**) malloc(nstacks*sizeof(stack_t));
   for (int istack=0; istack<nstacks; istack++) {
      stacklist[istack] = stacktree.stacks+istack;
   }

   qsort_stacklist(nstacks, stacklist);

   return stacklist;
}

void print_stack_branch(FILE *fp, int level, stacktree_t stacktree, int stackid) {
   // first print the indentation
   for (int ilevel=0; ilevel<level; ilevel++) {
      fprintf(fp, "  ");
   }
   stack_t stack = stacktree.stacks[stackid];
   fprintf(fp, "%s: id=%d\n",
           stack.name,
           stack.id);
   level++;
   for (int icallee=0; icallee<stack.ncallees; icallee++) {
      print_stack_branch(fp, level, stacktree, stack.callees[icallee]);
   }
}

void print_stacktree(FILE *fp, stacktree_t stacktree) {
   print_stack_branch(fp, 0, stacktree, 0);
}

int get_stack_string_length(stacktree_t stacktree, int stackid) {
   int stringlen = 0;
   int tmpstackid = stackid;
   stringlen += strlen(stacktree.stacks[stackid].name);
   stringlen ++; // function seperating character "<", or null terminator
   while (stacktree.stacks[tmpstackid].caller >= 0) {
      tmpstackid = stacktree.stacks[tmpstackid].caller;
      stringlen += strlen(stacktree.stacks[tmpstackid].name);
      stringlen ++; // function seperating character "<", or null terminator
   }
   return stringlen;
}

char *get_stack_string(stacktree_t stacktree, int stackid) {
   int stringlen = get_stack_string_length(stacktree, stackid);
   char *stackstring = (char*) malloc(stringlen*sizeof(char));
   // copy the chars one by one so there is no need to call strlen again.
   // thus minimizing reading the same memory locations over and over again.
   int tmpstackid = stackid;
   char *tmpname_ptr = stacktree.stacks[tmpstackid].name;
   char *tmpstackstring_ptr = stackstring;
   while (*tmpname_ptr != '\0') {
      *tmpstackstring_ptr = *tmpname_ptr;
      tmpstackstring_ptr++;
      tmpname_ptr++;
   }
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
   }
   // replace last char with a null terminator
   *tmpstackstring_ptr = '\0';
   return stackstring;
}

void print_stack(FILE *fp, stacktree_t stacktree, int stackid) {
   char *stackstr = get_stack_string(stacktree, stackid);
   fprintf(fp, " %8d", stacktree.stacks[stackid].ncalls);
   fprintf(fp, " %14.6lf", 1.0e-9*stacktree.stacks[stackid].time_nsec);
   fprintf(fp, " %14.6lf", 1.0e-9*stacktree.stacks[stackid].time_excl_nsec);
   fprintf(fp, " %s", stackstr);
   free(stackstr);
}

void print_stacklist(FILE *fp, stacktree_t stacktree) {
   fprintf(fp, "# ID    Calls    incl-time/s    excl-time/s Stack\n");
   for (int istack=0; istack<stacktree.nstacks; istack++) {
      fprintf(fp, "%4d", istack);
      print_stack(fp, stacktree, istack);
      fprintf(fp, "\n");
   }
}

void print_sorted_stacklist(FILE *fp, stack_t **sortedstacklist,
                            stacktree_t stacktree) {
   fprintf(fp, "# ID    Calls    incl-time/s    excl-time/s Stack\n");
   for (int istack=0; istack<stacktree.nstacks; istack++) {
      int stackid = sortedstacklist[istack]->id;
      fprintf(fp, "%4d", stackid);
      print_stack(fp, stacktree, stackid);
      fprintf(fp, "\n");
   }
}
