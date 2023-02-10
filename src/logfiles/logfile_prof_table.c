#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "self_profile.h"
#include "timer_types.h"
#include "table_types.h"
#include "configuration_types.h"
#include "collated_stack_types.h"
#include "vftrace_state.h"

#include "filenames.h"
#include "license.h"
#include "config.h"
#include "stacks.h"
#include "collate_stacks.h"
#include "tables.h"
#include "sorting.h"
#include "misc_utils.h"

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

double *vftr_logfile_prof_table_overhead_list(int nstacks, collated_stack_t **stack_ptrs) {
   double *overhead_list = (double*) malloc(nstacks*sizeof(double));

   for (int istack=0; istack<nstacks; istack++) {
      collated_stack_t *stack_ptr = stack_ptrs[istack];
      overhead_list[istack] = stack_ptr->profile.callprof.overhead_nsec;
      overhead_list[istack] *= 1.0e-9;
   }
   return overhead_list;
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

char **vftr_logfile_prof_table_stack_stackIDs_list(int nstacks, collated_stack_t **stack_ptrs, int max_stack_ids) {
   char **name_list = (char**) malloc(nstacks*sizeof(char*));
   for (int istack=0; istack<nstacks; istack++) {
      collated_stack_t *stack_ptr = stack_ptrs[istack];
      int strlen = 0;
      int cutoff_ids = stack_ptr->gid_list.ngids > max_stack_ids;
      int ngids = cutoff_ids ? max_stack_ids : stack_ptr->gid_list.ngids;
      // add up all the gids length in base 10
      for (int igid=0; igid<ngids; igid++) {
         strlen += vftr_count_base_digits(stack_ptr->gid_list.gids[igid], 10);
      }
      // add all the separating commata (not one less, because of the temporary null terminator)
      strlen += ngids;
      // add null terminator space
      strlen += 1;
      if (cutoff_ids) {
         // add ",..." space
         strlen += 4;
      }

      name_list[istack] = (char*) malloc(strlen*sizeof(char));
      char *tmpstr = name_list[istack];
      name_list[istack][0] = '\0';
      for (int igid=0; igid<ngids; igid++) {
         int ndigts = vftr_count_base_digits(stack_ptr->gid_list.gids[igid], 10);
         // plus two because ',' and '\0'
         snprintf(tmpstr, ndigts+2, "%d,", stack_ptr->gid_list.gids[igid]);
         tmpstr += ndigts+1;
      }
      if (cutoff_ids) {
         tmpstr--;
         snprintf(tmpstr, 5, ",...");
         tmpstr += 5;
      }
      tmpstr--;
      *tmpstr = '\0';
   }
   return name_list;
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
                                      config_t config) {
   SELF_PROFILE_START_FUNCTION;
   // first sort the stacktree according to the set config variables
   collated_stack_t **sorted_stacks =
      vftr_sort_collated_stacks_for_prof(config, stacktree);

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

   double *overhead_list = NULL;
   if (config.profile_table.show_overhead.value) {
      overhead_list = vftr_logfile_prof_table_overhead_list(stacktree.nstacks, sorted_stacks);
      vftr_table_add_column(&table, col_double, "overhead[s]", "%.3f", 'c', 'r', (void*) overhead_list);
   }

   double *imbalances_list = NULL;
   int *imbalance_ranks_list = NULL;
   if (config.profile_table.show_calltime_imbalances.value) {
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
   if (config.profile_table.show_callpath.value) {
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
   if (config.profile_table.show_overhead.value) {
      free(overhead_list);
   }
   if (config.profile_table.show_calltime_imbalances.value) {
      free(imbalances_list);
      free(imbalance_ranks_list);
   }
   free(function_names);
   free(caller_names);
   free(stack_IDs);
   if (config.profile_table.show_callpath.value) {
      for (int istack=0; istack<stacktree.nstacks; istack++) {
         free(path_list[istack]);
      }
      free(path_list);
   }

   free(sorted_stacks);
   SELF_PROFILE_END_FUNCTION;
}

void vftr_write_logfile_name_grouped_profile_table(FILE *fp,
                                                   collated_stacktree_t stacktree,
                                                   config_t config) {
   SELF_PROFILE_START_FUNCTION;
   // first sort the stacktree according to the set configuration variables
   collated_stack_t **sorted_stacks = vftr_sort_collated_stacks_for_prof(config, stacktree);

   fprintf(fp, "\nRuntime profile (Grouped by function name)\n");

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

   char **function_names = vftr_logfile_prof_table_stack_function_name_list(stacktree.nstacks, sorted_stacks);
   vftr_table_add_column(&table, col_string, "Function", "%s", 'c', 'r', (void*) function_names);

   char **stack_IDs = vftr_logfile_prof_table_stack_stackIDs_list(stacktree.nstacks, sorted_stacks,
                                                                  config.name_grouped_profile_table.max_stack_ids.value);
   vftr_table_add_column(&table, col_string, "STIDs", "%s", 'c', 'l', (void*) stack_IDs);

   vftr_print_table(fp, table);

   vftr_table_free(&table);
   free(calls);
   free(excl_time);
   free(excl_timer_perc);
   free(incl_time);
   free(function_names);
   for (int istack=0; istack<stacktree.nstacks; istack++) {
      free(stack_IDs[istack]);
   }
   free(stack_IDs);

   free(sorted_stacks);
   SELF_PROFILE_END_FUNCTION;
}
