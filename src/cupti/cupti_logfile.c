#include <stdio.h>

#include "symbols.h"
#include "tables.h"
#include "collated_stack_types.h"
#include "vftrace_state.h"

#include "cuptiprofiling_types.h"
#include "cupti_event_list.h"
#include "cupti_utils.h"

void vftr_get_total_cupti_times_for_logfile (collated_stacktree_t stacktree, 
                                             float *tot_compute_s, float *tot_memcpy_s, float *tot_other_s) {
   *tot_compute_s = 0;
   *tot_memcpy_s = 0;
   *tot_other_s = 0;

   for (int istack = 0; istack < stacktree.nstacks; istack++) {
      collated_stack_t this_stack = stacktree.stacks[istack];
      cupti_event_list_t *this_event = this_stack.profile.cuptiprof.events;
      while (this_event != NULL) {
         if (vftr_cupti_event_belongs_to_class (this_event, T_CUPTI_COMP)) *tot_compute_s += this_event->t_ms / 1000;
         if (vftr_cupti_event_belongs_to_class (this_event, T_CUPTI_MEMCP)) *tot_memcpy_s += this_event->t_ms / 1000;
         if (vftr_cupti_event_belongs_to_class (this_event, T_CUPTI_OTHER)) *tot_other_s += this_event->t_ms / 1000;
         this_event = this_event->next;
      }
   }
}

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
   size_t *memcpy_in = (size_t*)malloc(n_stackids_with_cupti_data*sizeof(size_t));
   size_t *memcpy_out = (size_t*)malloc(n_stackids_with_cupti_data*sizeof(size_t));
   char **names = (char**)malloc(n_stackids_with_cupti_data*sizeof(char*));
   char **callers = (char**)malloc(n_stackids_with_cupti_data*sizeof(char*));

   int i = 0;
   for (int istack = 0; istack < stacktree.nstacks; istack++) {
      collated_stack_t this_stack = stacktree.stacks[istack];
      cupti_event_list_t *this_event = this_stack.profile.cuptiprof.events;
      while (this_event != NULL) {
          stackids_with_cupti_data[i] = istack;
          calls[i] = this_event->n_calls;
          cbids[i] = this_event->cbid;
          t_compute[i] = vftr_cupti_event_belongs_to_class (this_event, T_CUPTI_COMP) ? this_event->t_ms / 1000 : 0;
          t_memcpy[i] = vftr_cupti_event_belongs_to_class (this_event, T_CUPTI_MEMCP) ? this_event->t_ms / 1000 : 0;
          t_other[i] = vftr_cupti_event_belongs_to_class (this_event, T_CUPTI_OTHER) ? this_event->t_ms / 1000 : 0;
          memcpy_in[i] = this_event->memcpy_bytes[CUPTI_COPY_IN];
          memcpy_out[i] = this_event->memcpy_bytes[CUPTI_COPY_OUT];
#ifdef _LIBERTY
          names[i] = vftr_demangle_cxx(this_event->func_name);
#else
          names[i] = this_event->func_name;
#endif
          callers[i] = this_stack.name;
          i++;
          this_event = this_event->next;
      }
   }

   table_t table = vftr_new_table();
   vftr_table_set_nrows(&table, n_stackids_with_cupti_data);

   vftr_table_add_column (&table, col_int, "SID", "%d", 'c', 'r', (void*)stackids_with_cupti_data);
   vftr_table_add_column (&table, col_string, "cudaName", "%s", 'c', 'r', (void*)names);
   vftr_table_add_column (&table, col_string, "Caller", "%s", 'c', 'r', (void*)callers);
   vftr_table_add_column (&table, col_int, "CBID", "%d", 'c', 'r', (void*)cbids);
   vftr_table_add_column (&table, col_int, "#Calls", "%d", 'c', 'r', (void*)calls);
   vftr_table_add_column (&table, col_float, "t_compute[s]", "%.2f", 'c', 'r', (void*)t_compute);
   vftr_table_add_column (&table, col_float, "t_memcpy[s]", "%.2f", 'c', 'r', (void*)t_memcpy);
   vftr_table_add_column (&table, col_float, "t_other[s]", "%.2f", 'c', 'r', (void*)t_other);
   vftr_table_add_column (&table, col_long, "Host->Device[B]", "%ld", 'c', 'r', (void*)memcpy_in);
   vftr_table_add_column (&table, col_long, "Device->Host[B]", "%ld", 'c', 'r', (void*)memcpy_out);

   fprintf (fp, "\n--Cuda Summary--\n");
   vftr_show_used_gpu_info (fp);
   fprintf (fp, "\n");
   vftr_print_table(fp, table); 

   vftr_table_free(&table);
   free(t_compute);
   free(t_memcpy);
   free(t_other);
   free(memcpy_in);
   free(memcpy_out);
   free(calls);
   free(cbids);
   free(names);
   free(callers);
   free(stackids_with_cupti_data);
}
