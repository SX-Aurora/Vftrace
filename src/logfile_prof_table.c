#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "timer_types.h"
#include "table_types.h"
#include "environment_types.h"
#include "stack_types.h"
#include "vftrace_state.h"

#include "filenames.h"
#include "license.h"
#include "config.h"
#include "environment.h"
#include "stacks.h"
#include "collate_stacks.h"
#include "tables.h"
#include "overheadprofiling_types.h"
#include "overheadprofiling.h"

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

int *vftr_stack_stackID_list(stacktree_t stacktree) {
   int nstacks = stacktree.nstacks;
   int *id_list = (int*) malloc(nstacks*sizeof(int));
   for (int istack=0; istack<nstacks; istack++) {
      id_list[istack] = stacktree.stacks[istack].gid;
   }
   return id_list;
}

void vftr_write_logfile_profile_table(FILE *fp, stacktree_t stacktree,
                                      environment_t environment) {
   fprintf(fp, "\nRuntime profile\n");

   table_t table = vftr_new_table();
   vftr_table_set_nrows(&table, stacktree.nstacks);

   int *calls = vftr_stack_calls_list(stacktree);
   vftr_table_add_column(&table, col_int, "Calls", "%d", 'c', 'r', (void*) calls);

   double *excl_time = vftr_stack_exclusive_time_list(stacktree);
   vftr_table_add_column(&table, col_double, "t_excl/s", "%.3f", 'c', 'r', (void*) excl_time);

   double *incl_time = vftr_stack_inclusive_time_list(stacktree);
   vftr_table_add_column(&table, col_double, "t_incl/s", "%.3f", 'c', 'r', (void*) incl_time);

  //
  // double *vftr_stack_overhead_time_list(int nstacks, stack_t *stacks);

   char **function_names = vftr_stack_function_name_list(stacktree);
   vftr_table_add_column(&table, col_string, "Function", "%s", 'c', 'r', (void*) function_names);

   char **caller_names = vftr_stack_caller_name_list(stacktree);
   vftr_table_add_column(&table, col_string, "Caller", "%s", 'c', 'r', (void*) caller_names);

   int *stack_IDs = vftr_stack_stackID_list(stacktree);
   vftr_table_add_column(&table, col_int, "ID", "%d", 'c', 'r', (void*) stack_IDs);

   vftr_print_table(fp, table);

   vftr_table_free(&table);
   free(calls);
   free(excl_time);
   free(incl_time);
   free(function_names);
   free(caller_names);
   free(stack_IDs);
}
