#include <stdio.h>
#include <stdlib.h>

#include "configuration_types.h"
#include "tables.h"
#include "callprofiling_types.h"
#include "stack_types.h"

#include "accprofiling_types.h"
#include "accprof_events.h"

void vftr_write_ranklogfile_accprof_table (FILE *fp, stacktree_t stacktree, config_t config) {
   int n_stackids_with_accprof_data = 0;
   
   for (int istack = 0; istack < stacktree.nstacks; istack++) {
      profile_t *this_profile = stacktree.stacks[istack].profiling.profiles;
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
   int *start_lines = (int*)malloc(n_stackids_with_accprof_data * sizeof(int));
   int *end_lines = (int*)malloc(n_stackids_with_accprof_data * sizeof(int));
   char **source_files = (char*)malloc(n_stackids_with_accprof_data * sizeof(char*));

   int i = 0;
   for (int istack = 0; istack < stacktree.nstacks; istack++) {
      stack_t this_stack = stacktree.stacks[istack];
      profile_t *this_profile = this_stack.profiling.profiles;
      accprofile_t accprof = this_profile->accprof;
      callprofile_t callprof = this_profile->callprof;
      acc_event_t ev = accprof.event_type;
      if (ev == 0) continue;
      stackids_with_accprof_data[i] = this_stack.gid;
      calls[i] = callprof.calls; 
      ev_names[i] = vftr_accprof_event_string(ev);
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
      start_lines[i] = accprof.line_start;
      end_lines[i] = accprof.line_end;
      source_files[i] = basename(accprof.source_file);
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
   vftr_table_add_column (&table, col_int, "Line 1", "%d", 'c', 'r', (void*)start_lines);
   vftr_table_add_column (&table, col_int, "Line 2", "%d", 'c', 'r', (void*)end_lines);

   fprintf (fp, "\n--OpenACC Summary--\n");
   vftr_print_table(fp, table);

   free (stackids_with_accprof_data);
   free (ev_names);
   free (calls);
   free (t_compute);
   free (t_memcpy);
   free (t_other);
   free (copied_bytes);
   free (start_lines);
   free (end_lines);
   free (source_files);
}
