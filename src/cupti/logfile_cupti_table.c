#include <stdio.h>

#include "tables.h"
#include "collated_stack_types.h"

#include "cuptiprofiling_types.h"
#include "cupti_event_list.h"

void vftr_write_logfile_cupti_table(FILE *fp, collated_stacktree_t stacktree) {

   int n_stackids_with_cupti_data = 0;
   for (int istack = 0; istack < stacktree.nstacks; istack++) {
      collated_stack_t this_stack = stacktree.stacks[istack];
      cupti_event_list_t *this_event = this_stack.profile.cuptiprof.events;
      while (this_event != NULL) {
          n_stackids_with_cupti_data++;
          this_event = this_event->next;
      } 
   }

   int *stackids_with_cupti_data = (int*)malloc(n_stackids_with_cupti_data*sizeof(int));
   int *calls = (int*)malloc(n_stackids_with_cupti_data*sizeof(int));
   int *cbids = (int*)malloc(n_stackids_with_cupti_data*sizeof(int));
   float *t_compute = (float*)malloc(n_stackids_with_cupti_data*sizeof(float));
   float *t_memcpy = (float*)malloc(n_stackids_with_cupti_data*sizeof(float));
   float *t_other = (float*)malloc(n_stackids_with_cupti_data*sizeof(float));
   char **names = (char**)malloc(n_stackids_with_cupti_data*sizeof(char*));
   char **callers = (char**)malloc(n_stackids_with_cupti_data*sizeof(char*));

   float total_t_compute = 0;
   float total_t_memcpy = 0;
   float total_t_other = 0;

   int i = 0;
   for (int istack = 0; istack < stacktree.nstacks; istack++) {
      collated_stack_t this_stack = stacktree.stacks[istack];
      cupti_event_list_t *this_event = this_stack.profile.cuptiprof.events;
      while (this_event != NULL) {
          stackids_with_cupti_data[i] = istack;
          calls[i] = this_event->n_calls;
          cbids[i] = this_event->cbid;
          t_compute[i] = cupti_event_belongs_to_class (this_event, T_CUPTI_COMP) ? this_event->t_ms / 1000 : 0;
          t_memcpy[i] = cupti_event_belongs_to_class (this_event, T_CUPTI_MEMCP) ? this_event->t_ms / 1000 : 0;
          t_other[i] = cupti_event_belongs_to_class (this_event, T_CUPTI_OTHER) ? this_event->t_ms / 1000 : 0;
          total_t_compute += t_compute[i];
          total_t_memcpy += t_memcpy[i];
          total_t_other += t_other[i];
          names[i] = this_event->func_name;
          callers[i] = this_stack.name;
          i++;
          this_event = this_event->next;
      }
   }

   table_t table = vftr_new_table();
   vftr_table_set_nrows(&table, n_stackids_with_cupti_data);

   vftr_table_add_column (&table, col_int, "SID", "%d", 'c', 'r', (void*)stackids_with_cupti_data);
   vftr_table_add_column (&table, col_string, "Caller", "%s", 'c', 'r', (void*)callers);
   vftr_table_add_column (&table, col_string, "cudaName", "%s", 'c', 'r', (void*)names);
   vftr_table_add_column (&table, col_int, "CBID", "%d", 'c', 'r', (void*)cbids);
   vftr_table_add_column (&table, col_int, "#Calls", "%d", 'c', 'r', (void*)calls);
   vftr_table_add_column (&table, col_float, "t_compute[s]", "%.2f", 'c', 'r', (void*)t_compute);
   vftr_table_add_column (&table, col_float, "t_memcpy[s]", "%.2f", 'c', 'r', (void*)t_memcpy);
   vftr_table_add_column (&table, col_float, "t_other[s]", "%.2f", 'c', 'r', (void*)t_other);
 
   fprintf (fp, "\n--CUDA summary--\n");
   fprintf (fp, "Total time: %.2f s\n", total_t_compute + total_t_memcpy + total_t_other);
   fprintf (fp, "   Compute: %.2f s\n", total_t_compute);
   fprintf (fp, "    Memcpy: %.2f s\n", total_t_memcpy);
   fprintf (fp, "     Other: %.2f s\n", total_t_other);
   fprintf (fp, "\n");
   vftr_print_table(fp, table); 

   vftr_table_free(&table);
   free(t_compute);
   free(t_memcpy);
   free(t_other);
   free(calls);
   free(cbids);
   free(names);
   free(callers);
   free(stackids_with_cupti_data);
}
