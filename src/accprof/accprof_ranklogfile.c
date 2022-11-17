#include <stdio.h>
#include <stdlib.h>

#include "configuration_types.h"
#include "tables.h"
#include "callprofiling_types.h"
#include "stack_types.h"
#include "sorting.h"

#include "accprofiling_types.h"
#include "accprof_events.h"

void vftr_get_total_accprof_times_for_ranklogfile (stacktree_t stacktree, double *tot_compute_s,
					           double *tot_memcpy_s, double *tot_other_s) {
   *tot_compute_s = 0;
   *tot_memcpy_s = 0;
   *tot_other_s = 0;

   for (int istack = 0; istack < stacktree.nstacks; istack++) {
      stack_t this_stack = stacktree.stacks[istack];
      accprofile_t accprof = this_stack.profiling.profiles[0].accprof;
      callprofile_t callprof = this_stack.profiling.profiles[0].callprof;
      if (vftr_accprof_is_data_event (accprof.event_type)) {
         *tot_memcpy_s += (double)callprof.time_excl_nsec / 1e9;
      } else if (vftr_accprof_is_launch_event (accprof.event_type)) {
         *tot_compute_s += (double)callprof.time_excl_nsec / 1e9;
      } else {
         *tot_other_s += (double)callprof.time_excl_nsec / 1e9;
      }
   }
}

void vftr_write_ranklogfile_accprof_table (FILE *fp, stacktree_t stacktree, config_t config) {
   int n_stackids_with_accprof_data = 0;
   
   stack_t **sorted_stacks = vftr_sort_stacks_for_accprof (config, stacktree);
   
   for (int istack = 0; istack < stacktree.nstacks; istack++) {
      profile_t *this_profile = sorted_stacks[istack]->profiling.profiles;
      accprofile_t accprof = this_profile->accprof;
      if (accprof.event_type != 0) n_stackids_with_accprof_data++;
   }


   int *stackids_with_accprof_data = (int*)malloc(n_stackids_with_accprof_data * sizeof(int));
   int *calls = (int*)malloc(n_stackids_with_accprof_data * sizeof(int));
   char **ev_names = (char**)malloc(n_stackids_with_accprof_data * sizeof(char*));
   size_t *copied_bytes = (size_t*)malloc(n_stackids_with_accprof_data * sizeof(size_t));
   double *t_compute = (double*)malloc(n_stackids_with_accprof_data * sizeof(double));
   double *t_memcpy = (double*)malloc(n_stackids_with_accprof_data * sizeof(double));
   double *t_other = (double*)malloc(n_stackids_with_accprof_data * sizeof(double));
   char **source_files = (char*)malloc(n_stackids_with_accprof_data * sizeof(char*));
   char **func_names = (char**)malloc(n_stackids_with_accprof_data * sizeof(char*));

   int i = 0;
   for (int istack = 0; istack < stacktree.nstacks; istack++) {
      stack_t *this_stack = sorted_stacks[istack];
      profile_t *this_profile = this_stack->profiling.profiles;
      accprofile_t accprof = this_profile->accprof;
      callprofile_t callprof = this_profile->callprof;
      acc_event_t ev = accprof.event_type;
      if (ev == 0) continue;
      stackids_with_accprof_data[i] = istack;
      calls[i] = callprof.calls; 
      if (ev != acc_ev_enqueue_launch_start && ev != acc_ev_enqueue_launch_end) {
          ev_names[i] = vftr_accprof_event_string(ev);
      } else {
          int slen = strlen(accprof.kernel_name) + 10;
	  ev_names[i] = (char*)malloc(slen * sizeof(char));
          snprintf (ev_names[i], slen, "launch (%s)", accprof.kernel_name);
      }
      copied_bytes[i] = accprof.copied_bytes;
      double t = (double)callprof.time_excl_nsec / 1e9;
      t_compute[i] = 0.0;
      t_memcpy[i] = 0.0;
      t_other[i] = 0.0;
      if (vftr_accprof_is_launch_event (ev)) {
         t_compute[i] = t;
      } else if (vftr_accprof_is_data_event (ev)) {
         t_memcpy[i] = t;
      } else {
         t_other[i] = t;
      }
      if (accprof.source_file != NULL) {
         source_files[i] = vftr_name_with_lines (basename(accprof.source_file), accprof.line_start, accprof.line_end);
      } else {
         source_files[i] = "unknown";
      }
      func_names[i] = accprof.func_name != NULL ? accprof.func_name : "unknown";
      i++;
   }

   table_t table = vftr_new_table();
   vftr_table_set_nrows(&table, n_stackids_with_accprof_data);

   vftr_table_add_column (&table, col_int, "STID", "%d", 'c', 'r', (void*)stackids_with_accprof_data);
   vftr_table_add_column (&table, col_string, "event", "%s", 'c', 'r', (void*)ev_names);
   vftr_table_add_column (&table, col_int, "#Calls", "%d", 'c', 'r', (void*)calls);
   vftr_table_add_column (&table, col_double, "t_compute[s]", "%.3lf", 'c', 'r', (void*)t_compute);
   vftr_table_add_column (&table, col_double, "t_memcpy[s]", "%.3lf", 'c', 'r', (void*)t_memcpy);
   vftr_table_add_column (&table, col_double, "t_other[s]", "%.3lf", 'c', 'r', (void*)t_other);
   vftr_table_add_column (&table, col_long, "Bytes", "%ld", 'c', 'r', (void*)copied_bytes);
   vftr_table_add_column (&table, col_string, "File", "%s", 'c', 'r', (void*)source_files);
   vftr_table_add_column (&table, col_string, "Function", "%s", 'c', 'r', (void*)func_names);

   fprintf (fp, "\n--OpenACC Summary--\n");
   vftr_print_table(fp, table);

   free (stackids_with_accprof_data);
   free (ev_names);
   free (calls);
   free (t_compute);
   free (t_memcpy);
   free (t_other);
   free (copied_bytes);
   free (source_files);
   free (func_names);
}
