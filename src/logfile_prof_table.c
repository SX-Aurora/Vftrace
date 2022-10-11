#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "self_profile.h"
#include "timer_types.h"
#include "table_types.h"
#include "environment_types.h"
#include "collated_stack_types.h"
#include "vftrace_state.h"

#include "filenames.h"
#include "license.h"
#include "config.h"
#include "environment.h"
#include "stacks.h"
#include "collate_stacks.h"
#include "tables.h"
#include "sorting.h"

int *vftr_logfile_prof_table_stack_calls_list(int nstacks, collated_stack_t **stack_ptrs) {
   int *calls_list = (int*) malloc(nstacks*sizeof(int));

   for (int istack=0; istack<nstacks; istack++) {
      collated_stack_t *stack_ptr = stack_ptrs[istack];
      calls_list[istack] = stack_ptr->profile.callprof.calls;
   }
   return calls_list;
}

double *vftr_logfile_prof_table_stack_inclusive_time_list(int nstacks, collated_stack_t **stack_ptrs) {
   double *inclusive_time_list = (double*) malloc(nstacks*sizeof(double));

   for (int istack=0; istack<nstacks; istack++) {
      collated_stack_t *stack_ptr = stack_ptrs[istack];
      inclusive_time_list[istack] = stack_ptr->profile.callprof.time_nsec;
      inclusive_time_list[istack] *= 1.0e-9;
   }
   return inclusive_time_list;
}

double *vftr_logfile_prof_table_stack_exclusive_time_list(int nstacks, collated_stack_t **stack_ptrs) {
   double *exclusive_time_list = (double*) malloc(nstacks*sizeof(double));

   for (int istack=0; istack<nstacks; istack++) {
      collated_stack_t *stack_ptr = stack_ptrs[istack];
      exclusive_time_list[istack] = stack_ptr->profile.callprof.time_excl_nsec;
      exclusive_time_list[istack] *= 1.0e-9;
   }
   return exclusive_time_list;
}

double *vftr_logfile_prof_table_stack_exclusive_time_percentage_list(int nstacks, collated_stack_t **stack_ptrs) {
   double *percent_list = (double*) malloc(nstacks*sizeof(double));
   long long total_time = 0ll;
   for (int istack=0; istack<nstacks; istack++) {
      collated_stack_t *stack_ptr = stack_ptrs[istack];
      total_time += stack_ptr->profile.callprof.time_excl_nsec;
   }
   double invtotal_time = 100.0/total_time;
   for (int istack=0; istack<nstacks; istack++) {
      collated_stack_t *stack_ptr = stack_ptrs[istack];
      percent_list[istack] = stack_ptr->profile.callprof.time_excl_nsec;
      percent_list[istack] *= invtotal_time;
   }
   return percent_list;
}

double *vftr_logfile_prof_table_imbalances_list(int nstacks, collated_stack_t **stack_ptrs) {
   double *imbalances_list = (double*) malloc(nstacks*sizeof(double));
   for (int istack=0; istack<nstacks; istack++) {
      collated_stack_t *stack_ptr = stack_ptrs[istack];
      imbalances_list[istack] = stack_ptr->profile.callprof.max_imbalance;
   }
   return imbalances_list;
}

int *vftr_logfile_prof_table_imbalance_ranks_list(int nstacks,
                                                  collated_stack_t **stack_ptrs) {
   int *imbalance_ranks_list = (int*) malloc(nstacks*sizeof(int));
   for (int istack=0; istack<nstacks; istack++) {
      collated_stack_t *stack_ptr = stack_ptrs[istack];
      imbalance_ranks_list[istack] = stack_ptr->profile.callprof.max_imbalance_on_rank;
   }
   return imbalance_ranks_list;
}

char **vftr_logfile_prof_table_stack_function_name_list(int nstacks, collated_stack_t **stack_ptrs) {
   char **name_list = (char**) malloc(nstacks*sizeof(char*));
   for (int istack=0; istack<nstacks; istack++) {
      collated_stack_t *stack_ptr = stack_ptrs[istack];
      name_list[istack] = stack_ptr->name;
   }
   return name_list;
}

char **vftr_logfile_prof_table_stack_caller_name_list(collated_stacktree_t stacktree, collated_stack_t **stack_ptrs) {
   int nstacks = stacktree.nstacks;
   char **name_list = (char**) malloc(nstacks*sizeof(char*));
   // the init function is never called
   for (int istack=0; istack<nstacks; istack++) {
      collated_stack_t *stack_ptr = stack_ptrs[istack];
      int callerID = stack_ptr->caller;
      if (callerID >= 0) {
         name_list[istack] = stacktree.stacks[callerID].name;
      } else {
         name_list[istack] = "----";
      }
   }
   return name_list;
}

int *vftr_logfile_prof_table_stack_stackID_list(int nstacks, collated_stack_t **stack_ptrs) {
   int *id_list = (int*) malloc(nstacks*sizeof(int));
   for (int istack=0; istack<nstacks; istack++) {
      collated_stack_t *stack_ptr = stack_ptrs[istack];
      id_list[istack] = stack_ptr->gid;
   }
   return id_list;
}

char **vftr_logfile_prof_table_callpath_list(int nstacks, collated_stack_t **stack_ptrs,
                                             collated_stacktree_t stacktree) {
   char **path_list = (char**) malloc(nstacks*sizeof(char*));
   for (int istack=0; istack<nstacks; istack++) {
      collated_stack_t *stack_ptr = stack_ptrs[istack];
      int stackid = stack_ptr->gid;
      path_list[istack] = vftr_get_collated_stack_string(stacktree, stackid, false);
   }
   return path_list;
}

void vftr_write_logfile_profile_table(FILE *fp, collated_stacktree_t stacktree,
                                      environment_t environment) {
   SELF_PROFILE_START_FUNCTION;
   // first sort the stacktree according to the set environment variables
   collated_stack_t **sorted_stacks = vftr_sort_collated_stacks_for_prof(environment, stacktree);

   fprintf(fp, "\nRuntime profile\n");

   table_t table = vftr_new_table();
   vftr_table_set_nrows(&table, stacktree.nstacks);

   int *calls = vftr_logfile_prof_table_stack_calls_list(stacktree.nstacks, sorted_stacks);
   vftr_table_add_column(&table, col_int, "Calls", "%d", 'c', 'r', (void*) calls);

   double *excl_time = vftr_logfile_prof_table_stack_exclusive_time_list(stacktree.nstacks, sorted_stacks);
   vftr_table_add_column(&table, col_double, "t_excl[s]", "%.3f", 'c', 'r', (void*) excl_time);

   double *excl_timer_perc = vftr_logfile_prof_table_stack_exclusive_time_percentage_list(stacktree.nstacks, sorted_stacks);
   vftr_table_add_column(&table, col_double, "t_excl[%]", "%.1f", 'c', 'r', (void*) excl_timer_perc);

   double *incl_time = vftr_logfile_prof_table_stack_inclusive_time_list(stacktree.nstacks, sorted_stacks);
   vftr_table_add_column(&table, col_double, "t_incl[s]", "%.3f", 'c', 'r', (void*) incl_time);

   double *imbalances_list = NULL;
   int *imbalance_ranks_list = NULL;
   if (environment.show_calltime_imbalances.value.bool_val) {
      imbalances_list = vftr_logfile_prof_table_imbalances_list(stacktree.nstacks, sorted_stacks);
      vftr_table_add_column(&table, col_double, "Imbalances[%]", "%6.2f", 'c', 'r', (void*) imbalances_list);

      imbalance_ranks_list = vftr_logfile_prof_table_imbalance_ranks_list(stacktree.nstacks, sorted_stacks);
      vftr_table_add_column(&table, col_int, "on rank", "%d", 'c', 'r', (void*) imbalance_ranks_list);
   }

   char **function_names = vftr_logfile_prof_table_stack_function_name_list(stacktree.nstacks, sorted_stacks);
   vftr_table_add_column(&table, col_string, "Function", "%s", 'c', 'r', (void*) function_names);

   char **caller_names = vftr_logfile_prof_table_stack_caller_name_list(stacktree, sorted_stacks);
   vftr_table_add_column(&table, col_string, "Caller", "%s", 'c', 'r', (void*) caller_names);

   int *stack_IDs = vftr_logfile_prof_table_stack_stackID_list(stacktree.nstacks, sorted_stacks);
   vftr_table_add_column(&table, col_int, "STID", "%d", 'c', 'r', (void*) stack_IDs);

   char **path_list = NULL;
   if (environment.callpath_in_profile.value.bool_val) {
      path_list = vftr_logfile_prof_table_callpath_list(stacktree.nstacks,
                                                        sorted_stacks,
                                                        stacktree);
      vftr_table_add_column(&table, col_string, "Callpath", "%s", 'c', 'r', (void*) path_list);
   }

   vftr_print_table(fp, table);

   vftr_table_free(&table);
   free(calls);
   free(excl_time);
   free(excl_timer_perc);
   free(incl_time);
   if (environment.show_calltime_imbalances.value.bool_val) {
      free(imbalances_list);
      free(imbalance_ranks_list);
   }
   free(function_names);
   free(caller_names);
   free(stack_IDs);
   if (environment.callpath_in_profile.value.bool_val) {
      for (int istack=0; istack<stacktree.nstacks; istack++) {
         free(path_list[istack]);
      }
      free(path_list);
   }

   free(sorted_stacks);
   SELF_PROFILE_END_FUNCTION;
}
