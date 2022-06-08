#include <stdlib.h>

#include "stack_types.h"
#include "table_types.h"

long long vftr_total_overhead_usec(stacktree_t stacktree) {
   long long overhead = 0;
   for (int istack=0; istack<stacktree.nstacks; istack++) {
      stack_t *stack_ptr = stacktree.stacks + istack;
      for (int iprof=0; iprof<stack_ptr->profiling.nprofiles; iprof++) {
         profile_t *prof_ptr = stack_ptr->profiling.profiles+iprof;
         overhead += prof_ptr->overheadProf.hook_usec;
         // TODO: Add MPI and OMP overhead
      }
   }
   return overhead;
}

int *vftr_stack_calls_list(stacktree_t stacktree) {
   int nstacks = stacktree.nstacks;
   int *calls_list = (int*) malloc(nstacks*sizeof(int));

   for (int istack=0; istack<nstacks; istack++) {
      stack_t *stack_ptr = stacktree.stacks + istack;
      calls_list[istack] = 0;
      for (int iprof=0; iprof<stack_ptr->profiling.nprofiles; iprof++) {
         profile_t *prof_ptr = stack_ptr->profiling.profiles+iprof;
         calls_list[istack] += prof_ptr->callProf.calls;
      }
   }
   return calls_list;
}

double *vftr_stack_inclusive_time_list(stacktree_t stacktree) {
   int nstacks = stacktree.nstacks;
   double *inclusive_time_list = (double*) malloc(nstacks*sizeof(double));

   for (int istack=0; istack<nstacks; istack++) {
      stack_t *stack_ptr = stacktree.stacks + istack;
      profile_t *prof_ptr = stack_ptr->profiling.profiles;
      inclusive_time_list[istack] = prof_ptr->callProf.time_usec;
      inclusive_time_list[istack] *= 1.0e-6;
   }
   // the init function has as inclusive time the inclusive time of main
   inclusive_time_list[0] = inclusive_time_list[1];
   return inclusive_time_list;
}

double *vftr_stack_exclusive_time_list(stacktree_t stacktree) {
   int nstacks = stacktree.nstacks;
   double *exclusive_time_list = (double*) malloc(nstacks*sizeof(double));

   for (int istack=0; istack<nstacks; istack++) {
      stack_t *stack_ptr = stacktree.stacks + istack;
      profile_t *prof_ptr = stack_ptr->profiling.profiles;
      exclusive_time_list[istack] = prof_ptr->callProf.time_excl_usec;
      exclusive_time_list[istack] *= 1.0e-6;
   }
   return exclusive_time_list;
}

double *vftr_stack_overhead_time_list(stacktree_t stacktree) {
   int nstacks = stacktree.nstacks;
   double *overhead_time_list = (double*) malloc(nstacks*sizeof(double));

   for (int istack=0; istack<nstacks; istack++) {
      stack_t *stack_ptr = stacktree.stacks + istack;
      profile_t *prof_ptr = stack_ptr->profiling.profiles;
      overhead_time_list[istack] = prof_ptr->overheadProf.hook_usec;
      overhead_time_list[istack] *= 1.0e-6;
   }
   return overhead_time_list;
}

char **vftr_stack_function_name_list(stacktree_t stacktree) {
   int nstacks = stacktree.nstacks;
   char **name_list = (char**) malloc(nstacks*sizeof(char*));
   for (int istack=0; istack<nstacks; istack++) {
      name_list[istack] = stacktree.stacks[istack].name;
   }
   return name_list;
}

char **vftr_stack_caller_name_list(stacktree_t stacktree) {
   int nstacks = stacktree.nstacks;
   char **name_list = (char**) malloc(nstacks*sizeof(char*));
   // the init function is never called
   name_list[0] = "----";
   for (int istack=1; istack<nstacks; istack++) {
      int callerID = stacktree.stacks[istack].caller;
      name_list[istack] = stacktree.stacks[callerID].name;
   }
   return name_list;
}
