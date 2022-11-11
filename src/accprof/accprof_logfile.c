#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "configuration_types.h"
#include "collated_stack_types.h"
#include "collated_profiling_types.h"
#include "callprofiling_types.h"
#include "tables.h"
#include "misc_utils.h"
#include "sorting.h"

#include "accprofiling_types.h"
#include "accprof_events.h"

void vftr_get_total_accprof_times_for_logfile (collated_stacktree_t stacktree,
					       double *tot_compute_s, double *tot_memcpy_s, double *tot_other_s) {
   *tot_compute_s = 0;
   *tot_memcpy_s = 0;
   *tot_other_s = 0;

   for (int istack = 0; istack < stacktree.nstacks; istack++) {
      collated_stack_t this_stack = stacktree.stacks[istack];
      collated_callprofile_t callprof = this_stack.profile.callprof;
      accprofile_t accprof = this_stack.profile.accprof;
      if (vftr_accprof_is_data_event (accprof.event_type)) {
         *tot_memcpy_s += (double)callprof.time_excl_nsec / 1e9;
      } else if (vftr_accprof_is_launch_event (accprof.event_type)) {
         *tot_compute_s += (double)callprof.time_excl_nsec / 1e9;
      } else {
         *tot_other_s = (double)callprof.time_excl_nsec / 1e9;
      }
   }
}

char *vftr_name_with_lines (char *name, int line_1, int line_2) {
   int n1 = strlen(name);
   int n2 = vftr_count_base_digits ((long long)line_1, 10);
   int n3 = vftr_count_base_digits ((long long)line_2, 10);
   int slen = n1 + n2 + n3 + 7;
   char *s = (char*) malloc (slen * sizeof(char));
   snprintf (s, slen, "%s (%d - %d)", name, line_1, line_2); 
   return s;
}

void vftr_write_logfile_accprof_table (FILE *fp, collated_stacktree_t stacktree, config_t config) {
   int n_stackids_with_accprof_data = 0;

   collated_stack_t **sorted_stacks = vftr_sort_collated_stacks_for_accprof (config, stacktree);

   for (int istack = 0; istack < stacktree.nstacks; istack++) {
      accprofile_t accprof = sorted_stacks[istack]->profile.accprof;
      if (accprof.event_type != 0) n_stackids_with_accprof_data++;
   }


   int *stackids_with_accprof_data = (int*)malloc(n_stackids_with_accprof_data * sizeof(int));
   int *calls = (int*)malloc(n_stackids_with_accprof_data * sizeof(int));
   char **ev_names = (char**)malloc(n_stackids_with_accprof_data * sizeof(char*));
   size_t *copied_bytes = (size_t*)malloc(n_stackids_with_accprof_data * sizeof(size_t));
   double *t_compute = (double*)malloc(n_stackids_with_accprof_data * sizeof(double));
   double *t_memcpy = (double*)malloc(n_stackids_with_accprof_data * sizeof(double));
   double *t_other = (double*)malloc(n_stackids_with_accprof_data * sizeof(double));
   char **source_files = (char**)malloc(n_stackids_with_accprof_data * sizeof(char*));
   char **func_names = (char**)malloc(n_stackids_with_accprof_data * sizeof(char*));

   int i = 0;
   for (int istack = 0; istack < stacktree.nstacks; istack++) {
      collated_stack_t *this_stack = sorted_stacks[istack];
      accprofile_t accprof = this_stack->profile.accprof;
      collated_callprofile_t callprof = this_stack->profile.callprof;
      acc_event_t ev = accprof.event_type;
      if (ev == 0) continue;
      stackids_with_accprof_data[i] = this_stack->gid;
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
      source_files[i] = vftr_name_with_lines (basename(accprof.source_file), accprof.line_start, accprof.line_end);
      //func_names[i] = vftr_name_with_lines (accprof.func_name, accprof.func_line_start, accprof.func_line_end);
      func_names[i] = accprof.func_name;
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
   vftr_table_add_column (&table, col_string ,"Source File", "%s", 'c', 'r', (void*)source_files);
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

void vftr_write_logfile_accev_names (FILE *fp) {
   fprintf (fp, "acc_ev_none: %d\n", acc_ev_none); 
   fprintf (fp, "acc_ev_device_init_start: %d\n", acc_ev_device_init_start); 
   fprintf (fp, "acc_ev_device_init_end: %d\n", acc_ev_device_init_end); 
   fprintf (fp, "acc_ev_device_shutdown_start: %d\n", acc_ev_device_shutdown_start); 
   fprintf (fp, "acc_ev_device_shutdown_end: %d\n", acc_ev_device_shutdown_end); 
   fprintf (fp, "acc_ev_runtime_shutdown: %d\n", acc_ev_runtime_shutdown); 
   fprintf (fp, "acc_ev_create: %d\n", acc_ev_create);
   fprintf (fp, "acc_ev_delete: %d\n", acc_ev_delete);
   fprintf (fp, "acc_ev_alloc: %d\n", acc_ev_alloc);
   fprintf (fp, "acc_ev_free: %d\n", acc_ev_free);
   fprintf (fp, "acc_ev_enter_data_start: %d\n", acc_ev_enter_data_start);
   fprintf (fp, "acc_ev_enter_data_end: %d\n", acc_ev_enter_data_end);
   fprintf (fp, "acc_ev_exit_data_start: %d\n", acc_ev_exit_data_start);
   fprintf (fp, "acc_ev_exit_data_end: %d\n", acc_ev_exit_data_end);
   fprintf (fp, "acc_ev_update_start: %d\n", acc_ev_update_start);
   fprintf (fp, "acc_ev_update_end: %d\n", acc_ev_update_end);
   fprintf (fp, "acc_ev_compute_construct_start: %d\n", acc_ev_compute_construct_start);
   fprintf (fp, "acc_ev_compute_construct_end: %d\n", acc_ev_compute_construct_end);
   fprintf (fp, "acc_ev_enqueue_launch_start: %d\n", acc_ev_enqueue_launch_start);
   fprintf (fp, "acc_ev_enqueue_launch_end: %d\n", acc_ev_enqueue_launch_end);
   fprintf (fp, "acc_ev_enqueue_upload_start: %d\n", acc_ev_enqueue_upload_start);
   fprintf (fp, "acc_ev_enqueue_upload_end: %d\n", acc_ev_enqueue_upload_end);
   fprintf (fp, "acc_ev_enqueue_download_start: %d\n", acc_ev_enqueue_download_start);
   fprintf (fp, "acc_ev_enqueue_download_end: %d\n", acc_ev_enqueue_download_end);
   fprintf (fp, "acc_ev_wait_start: %d\n", acc_ev_wait_start);
   fprintf (fp, "acc_ev_wait_end: %d\n", acc_ev_wait_end);
   fprintf (fp, "acc_ev_last: %d\n", acc_ev_last);
  



}
