#include <stdio.h>
#include <stdlib.h>
#include <limits.h>

#include "vftrace_state.h"
#include "table_types.h"
#include "tables.h"
#include "sorting.h"
#include "collated_stack_types.h"

void vftr_write_minmax_summary (FILE *fp, vftrace_t vftrace) {
   collated_stack_t **sorted_stacks =
      vftr_sort_collated_stacks_tmax(vftrace.config, vftrace.process.collated_stacktree);

   int nstacks = vftrace.process.collated_stacktree.nstacks;
   table_t table = vftr_new_table();
   vftr_table_set_nrows(&table, nstacks);      

   int *stack_ids = (int*)malloc(nstacks * sizeof(int));
   char **function_names = (char**)malloc(nstacks * sizeof(char*)); 
   double *t_max_s = (double*)malloc(nstacks * sizeof(double));
   double *t_min_s = (double*)malloc(nstacks * sizeof(double));
   int *rank_max = (int*)malloc(nstacks * sizeof(int));
   int *rank_min = (int*)malloc(nstacks * sizeof(int));
   for (int i = 0; i < nstacks; i++) {
      stack_ids[i] = i;
      collated_stack_t *this_stack = sorted_stacks[i];
      function_names[i] = this_stack->name;
      t_max_s[i] = (double)this_stack->profile.callprof.max_time_nsec / 1e9;
      t_min_s[i] = (double)this_stack->profile.callprof.min_time_nsec / 1e9;
      rank_max[i] = this_stack->profile.callprof.max_on_rank;
      rank_min[i] = this_stack->profile.callprof.min_on_rank;
   }

   vftr_table_add_column (&table, col_int, "STID", "%d", 'c', 'r', (void*)stack_ids);
   vftr_table_add_column (&table, col_string, "Function", "%s", 'c', 'r', (void*)function_names);
   vftr_table_add_column (&table, col_double, "t_max[s]", "%.3f", 'c', 'r', (void*)t_max_s);
   vftr_table_add_column (&table, col_int, "rank", "%d", 'c', 'r', (void*)rank_max);
   vftr_table_add_column (&table, col_double, "t_min[s]", "%.3f", 'c', 'r', (void*)t_min_s);
   vftr_table_add_column (&table, col_int, "rank", "%d", 'c', 'r', (void*)rank_min);

   fprintf (fp, "\n\n Summary: Min/Max runtime across all ranks\n");
   vftr_print_table (fp, table);

   free(stack_ids);
   free(function_names);
   free(t_max_s);   
   free(t_min_s);
   free(rank_max);
   free(rank_min);
}
