#include <stdlib.h>
#include <stdio.h>

#include <string.h>

#include "self_profile.h"
#include "timer_types.h"
#include "table_types.h"
#include "configuration_types.h"
#include "collated_stack_types.h"
#include "vftrace_state.h"
#include "vedaprofiling_types.h"

#include "config.h"
#include "stacks.h"
#include "collate_stacks.h"
#include "tables.h"
#include "sorting.h"

int vftr_logfile_veda_table_nrows(collated_stacktree_t stacktree) {
   int nstacks = stacktree.nstacks;
   int nrows = 0;
   for (int istack=0; istack<nstacks; istack++) {
      collated_stack_t *stack_ptr = stacktree.stacks+istack;
      vedaprofile_t vedaprof = stack_ptr->profile.vedaprof;
      if (vedaprof.n_calls > 0) {
         nrows++;
      }
   }
   return nrows;
}

collated_stack_t **vftr_logfile_veda_table_get_relevant_collated_stacks(
   int nrows, collated_stacktree_t stacktree) {

   collated_stack_t **selected_stacks = (collated_stack_t**)
      malloc(nrows*sizeof(collated_stack_t));
   int irow = 0;
   for (int istack=0; istack<stacktree.nstacks; istack++) {
      collated_stack_t *stack_ptr = stacktree.stacks+istack;
      vedaprofile_t vedaprof = stack_ptr->profile.vedaprof;
      if (vedaprof.n_calls > 0) {
         selected_stacks[irow] = stack_ptr;
         irow++;
      }
   }
   return selected_stacks;
}

void vftr_write_logfile_veda_table(FILE *fp, collated_stacktree_t stacktree,
                                   config_t config) {
   SELF_PROFILE_START_FUNCTION;
   int nrows = vftr_logfile_veda_table_nrows(stacktree);
   collated_stack_t **selected_stacks =
      vftr_logfile_veda_table_get_relevant_collated_stacks(nrows, stacktree);
   //TODO vftr_sort_collated_stacks_for_vedaprof(config, nrows, selected_stacks);

   fprintf(fp, "\nVEDA profile\n");
   table_t table = vftr_new_table();
   vftr_table_set_nrows(&table, nrows);

   int *ncalls = vftr_logfile_veda_table_ncalls_list(nrows, selected_stacks);
   vftr_table_add_column(&table, col_int, "ncalls", "%d", 'c', 'r', (void*) nmessages);

   vftr_print_table(fp, table);
   vftr_table_free(&table);
   free(ncalls);

   free(selected_stacks);
   SELF_PROFILE_END_FUNCTION;
} 
