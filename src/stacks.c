#include <stdlib.h>
#include <stdbool.h>

#include <string.h>

#include "self_profile.h"
#include "stack_types.h"
#include "collated_stack_types.h"
#include "realloc_consts.h"
#include "stacks.h"
#include "profiling.h"
#include "callprofiling.h"
#include "symbols.h"
#include "hashing.h"
#include "search.h"
#include "collate_stacks.h"
#include "hwprofiling.h"

#ifdef _ACCPROF
#include "accprof_events.h"
#endif

void vftr_stacktree_realloc(stacktree_t *stacktree_ptr) {
   SELF_PROFILE_START_FUNCTION;
   stacktree_t stacktree = *stacktree_ptr;
   while (stacktree.nstacks > stacktree.maxstacks) {
      int maxstacks = stacktree.maxstacks*vftr_realloc_rate+vftr_realloc_add;
      stacktree.stacks = (vftr_stack_t*)
         realloc(stacktree.stacks, maxstacks*sizeof(vftr_stack_t));
      stacktree.maxstacks = maxstacks;
   }
   *stacktree_ptr = stacktree;
   SELF_PROFILE_END_FUNCTION;
}

void vftr_stack_callees_realloc(vftr_stack_t *stack_ptr) {
   SELF_PROFILE_START_FUNCTION;
   vftr_stack_t stack = *stack_ptr;
   while (stack.ncallees > stack.maxcallees) {
      int maxcallees = stack.maxcallees*vftr_realloc_rate+vftr_realloc_add;
      stack.callees = (int*)
         realloc(stack.callees, maxcallees*sizeof(int));
      stack.maxcallees = maxcallees;
   }
   *stack_ptr = stack;
   SELF_PROFILE_END_FUNCTION;
}

void vftr_insert_callee(int calleeID, vftr_stack_t *caller) {
   SELF_PROFILE_START_FUNCTION;
   // append the callee to the list
   int idx = caller->ncallees;
   caller->ncallees++;
   vftr_stack_callees_realloc(caller);
   caller->callees[idx] = calleeID;
   SELF_PROFILE_END_FUNCTION;
}

int vftr_new_stack(int callerID, stacktree_t *stacktree_ptr,
                   const char *name, const char *cleanname,
                   uintptr_t address, bool precise) {
   SELF_PROFILE_START_FUNCTION;
   int stackID = stacktree_ptr->nstacks;
   stacktree_ptr->nstacks++;
   vftr_stacktree_realloc(stacktree_ptr);

   vftr_stack_t *callerstack = stacktree_ptr->stacks+callerID;

   vftr_stack_t *stack = stacktree_ptr->stacks+stackID;
   stack->address = address;
   stack->precise = precise;
   stack->caller = callerstack->lid;
   stack->maxcallees = 0;
   stack->ncallees = 0;
   stack->callees = NULL;
   stack->lid = stackID;

   stack->name = strdup(name);
   stack->cleanname = strdup(cleanname);

   stack->profiling = vftr_new_profilelist();
   vftr_insert_callee(stack->lid, callerstack);

   SELF_PROFILE_END_FUNCTION;
   return stack->lid;
}

vftr_stack_t vftr_first_stack() {
   SELF_PROFILE_START_FUNCTION;
   vftr_stack_t stack;
   stack.address = (uintptr_t) NULL;
   stack.precise = false;
   stack.caller = -1;
   stack.maxcallees = 0;
   stack.ncallees = 0;
   stack.callees = NULL;
   stack.lid = 0;
   stack.gid = 0;
   stack.name = strdup("init");
   stack.cleanname = strdup(stack.name);
   stack.profiling = vftr_new_profilelist();
   vftr_new_profile_in_list(0, &(stack.profiling));
   stack.hash = 0;
   SELF_PROFILE_END_FUNCTION;
   return stack;
}

// recursively free the stack tree
void vftr_stack_free(vftr_stack_t *stacks_ptr, int stackID) {
   vftr_stack_t stack = stacks_ptr[stackID];
   if (stack.ncallees > 0) {
      for (int icallee=0; icallee<stack.ncallees; icallee++) {
         vftr_stack_free(stacks_ptr, stack.callees[icallee]);
      }
      free(stack.callees);
      stack.callees = NULL;
   }
   vftr_profilelist_free(&(stack.profiling));
   free(stack.name);
   free(stack.cleanname);
   stacks_ptr[stackID] = stack;
}

stacktree_t vftr_new_stacktree() {
   SELF_PROFILE_START_FUNCTION;
   stacktree_t stacktree;
   stacktree.maxstacks = 0;
   stacktree.stacks = NULL;
   stacktree.nstacks = 1;
   vftr_stacktree_realloc(&stacktree);
   stacktree.stacks[0] = vftr_first_stack();
   SELF_PROFILE_END_FUNCTION;
   return stacktree;
}

void vftr_stacktree_free(stacktree_t *stacktree_ptr) {
   SELF_PROFILE_START_FUNCTION;
   stacktree_t stacktree = *stacktree_ptr;
   if (stacktree.nstacks > 0) {
      vftr_stack_free(stacktree.stacks, 0);
      free(stacktree.stacks);
      stacktree.stacks = NULL;
      stacktree.nstacks = 0;
      stacktree.maxstacks = 0;
   }
   *stacktree_ptr = stacktree;
   SELF_PROFILE_END_FUNCTION;
}

// fill in data that was not computed during runtime
void vftr_finalize_stacktree(stacktree_t *stacktree_ptr) {
   SELF_PROFILE_START_FUNCTION;
   // exclusive time
   vftr_update_stacks_exclusive_time(stacktree_ptr);
   vftr_update_stacks_exclusive_counters(stacktree_ptr);
   vftr_update_stacks_hw_observables(stacktree_ptr);
   SELF_PROFILE_END_FUNCTION;
}

void vftr_print_stack_branch(FILE *fp, int level, stacktree_t stacktree, int stackid) {
   // first print the indentation
   for (int ilevel=0; ilevel<level; ilevel++) {
      fprintf(fp, "  ");
   }
   vftr_stack_t stack = stacktree.stacks[stackid];
   fprintf(fp, "%s (%llx): id=%d\n",
           stack.name,
           (unsigned long long) stack.address,
           stack.lid);
   level++;
   for (int icallee=0; icallee<stack.ncallees; icallee++) {
      vftr_print_stack_branch(fp, level, stacktree, stack.callees[icallee]);
   }
}

void vftr_print_stacktree(FILE *fp, stacktree_t stacktree) {
   vftr_print_stack_branch(fp, 0, stacktree, 0);
}

int vftr_stack_string_entry_length (vftr_stack_t stack) {
   int stringlen = strlen(stack.cleanname);
#ifdef _ACCPROF
   accprofile_t accprof = stack.profiling.profiles[0].accprof;
   if (vftr_accprof_event_is_defined (accprof.event_type)) {
      char *event_string = vftr_accprof_event_string (accprof.event_type);
      stringlen += 6 + strlen(event_string);
   }
#endif
   return stringlen + 1; // Add 1 for function seperating character "<", or null terminator
}

void vftr_fill_stack_string_entry (char **stackstring_ptr, vftr_stack_t stack) {
   char *tmpname_ptr = stack.cleanname;
   while (*tmpname_ptr != '\0') {
      **stackstring_ptr = *tmpname_ptr;
      (*stackstring_ptr)++;
      tmpname_ptr++;
   }
#ifdef _ACCPROF
   accprofile_t accprof = stack.profiling.profiles[0].accprof;
   if (vftr_accprof_event_is_defined (accprof.event_type)) {
      char *event_string = vftr_accprof_event_string (accprof.event_type);
      tmpname_ptr = "(ACC:";
      while (*tmpname_ptr != '\0') {
         **stackstring_ptr = *tmpname_ptr;
         (*stackstring_ptr)++;
         tmpname_ptr++;
      }
      tmpname_ptr = event_string;
      while (*tmpname_ptr != '\0') {
         **stackstring_ptr = *tmpname_ptr;
         (*stackstring_ptr)++;
         tmpname_ptr++;
      }
      **stackstring_ptr = ')';
      (*stackstring_ptr)++;
   }
#endif
}

int vftr_get_stack_string_length(stacktree_t stacktree, int stackid, bool show_precise) {
   int stringlen = 0;
   int tmpstackid = stackid;
   stringlen += vftr_stack_string_entry_length (stacktree.stacks[stackid]);
   while (stacktree.stacks[tmpstackid].caller >= 0) {
      tmpstackid = stacktree.stacks[tmpstackid].caller;
      stringlen += vftr_stack_string_entry_length (stacktree.stacks[tmpstackid]);
      if (show_precise && stacktree.stacks[tmpstackid].precise) {
         stringlen ++; // '*' for indicating precise functions
      }
   }
   return stringlen;
}

char *vftr_get_stack_string(stacktree_t stacktree, int stackid, bool show_precise) {
   int stringlen = vftr_get_stack_string_length(stacktree, stackid, show_precise);
   char *stackstring = (char*) malloc(stringlen*sizeof(char));
   // copy the chars one by one so there is no need to call strlen again.
   // thus minimizing reading the same memory locations over and over again.
   int tmpstackid = stackid;
   char *tmpname_ptr = stacktree.stacks[tmpstackid].cleanname;
   char *tmpstackstring_ptr = stackstring;
   vftr_fill_stack_string_entry (&tmpstackstring_ptr, stacktree.stacks[tmpstackid]);
   if (show_precise && stacktree.stacks[tmpstackid].precise) {
      *tmpstackstring_ptr = '*';
      tmpstackstring_ptr++;
   }
   while (stacktree.stacks[tmpstackid].caller >= 0) {
      // add function name separating character
      *tmpstackstring_ptr = '<';
      tmpstackstring_ptr++;
      tmpstackid = stacktree.stacks[tmpstackid].caller;
      vftr_fill_stack_string_entry (&tmpstackstring_ptr, stacktree.stacks[tmpstackid]);
      if (show_precise && stacktree.stacks[tmpstackid].precise) {
         *tmpstackstring_ptr = '*';
         tmpstackstring_ptr++;
      }
   }
   // replace last char with a null terminator
   *tmpstackstring_ptr = '\0';
   return stackstring;
}

void vftr_print_stack(FILE *fp, stacktree_t stacktree, int stackid) {
   char *stackstr = vftr_get_stack_string(stacktree, stackid, false);
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
