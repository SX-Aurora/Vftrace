#include <stdlib.h>

#include "stack_types.h"
#include "table_types.h"

long long vftr_total_overhead_usec(stacktree_t stacktree) {
   long long overhead = 0;
   for (int istack=0; istack<stacktree.nstacks; istack++) {
      overhead += stacktree.stacks[istack].profiling.callProf.overhead_time_usec;
   }
   return overhead;
}

int *vftr_stack_calls_list(int nstacks, stack_t *stacks) {
   int *calls_list = (int*) malloc(nstacks*sizeof(int));

   for (int istack=0; istack<nstacks; istack++) {
      calls_list[istack] = stacks[istack].profiling.callProf.calls;
   }
   return calls_list;
}

double *vftr_stack_inclusive_time_list(int nstacks, stack_t *stacks) {
   double *inclusive_time_list = (double*) malloc(nstacks*sizeof(double));

   for (int istack=0; istack<nstacks; istack++) {
      inclusive_time_list[istack] = stacks[istack].profiling.callProf.time_usec;
      inclusive_time_list[istack] *= 1.0e-6;
   }
   // the init function has as inclusive time the inclusive time of main
   inclusive_time_list[0] = stacks[1].profiling.callProf.time_usec * 1.0e-6;
   return inclusive_time_list;
}

double *vftr_stack_exclusive_time_list(int nstacks, stack_t *stacks) {
   double *exclusive_time_list = (double*) malloc(nstacks*sizeof(double));

   for (int istack=0; istack<nstacks; istack++) {
      exclusive_time_list[istack] = stacks[istack].profiling.callProf.time_excl_usec;
      exclusive_time_list[istack] *= 1.0e-6;
   }
   return exclusive_time_list;
}

double *vftr_stack_overhead_time_list(int nstacks, stack_t *stacks) {
   double *overhead_time_list = (double*) malloc(nstacks*sizeof(double));

   for (int istack=0; istack<nstacks; istack++) {
      overhead_time_list[istack] = stacks[istack].profiling.callProf.overhead_time_usec;
      overhead_time_list[istack] *= 1.0e-6;
   }
   return overhead_time_list;
}

char **vftr_stack_function_name_list(int nstacks, stack_t *stacks) {
   char **name_list = (char**) malloc(nstacks*sizeof(char*));
   for (int istack=0; istack<nstacks; istack++) {
      name_list[istack] = stacks[istack].name;
   }
   return name_list;
}

char **vftr_stack_caller_name_list(int nstacks, stack_t *stacks) {
   char **name_list = (char**) malloc(nstacks*sizeof(char*));
   // the init function is never called
   name_list[0] = "----";
   for (int istack=1; istack<nstacks; istack++) {
      int callerID = stacks[istack].caller;
      name_list[istack] = stacks[callerID].name;
   }
   return name_list;
}
