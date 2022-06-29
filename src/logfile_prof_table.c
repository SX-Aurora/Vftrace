#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "timer_types.h"
#include "table_types.h"
#include "environment_types.h"
#include "vftrace_state.h"

#include "filenames.h"
#include "license.h"
#include "config.h"
#include "log_profile.h"
#include "environment.h"
#include "stacks.h"
#include "collate_stacks.h"
#include "tables.h"
#include "overheadprofiling_types.h"
#include "overheadprofiling.h"

void vftr_write_logfile_profile_table(FILE *fp, stacktree_t stacktree,
                                      environment_t environment,
                                      long long runtime) {
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

   vftr_print_table(fp, table);

   vftr_table_free(&table);
   free(calls);
   free(excl_time);
   free(incl_time);
   free(function_names);
   free(caller_names);
}
