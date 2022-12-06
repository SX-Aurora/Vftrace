#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <libgen.h>

#include "configuration_types.h"
#include "tables.h"
#include "callprofiling_types.h"
#include "stack_types.h"
#include "sorting.h"

#include "accprof_init_final.h"
#include "accprofiling_types.h"
#include "accprof_events.h"
#include "accprof_logfile.h"

void vftr_get_total_accprof_times_for_ranklogfile (stacktree_t stacktree, double *tot_compute_s,
					           double *tot_data_s, double *tot_wait_s) {
   *tot_compute_s = 0;
   *tot_data_s = 0;
   *tot_wait_s = 0;

   for (int istack = 0; istack < stacktree.nstacks; istack++) {
      stack_t this_stack = stacktree.stacks[istack];
      accprofile_t accprof = this_stack.profiling.profiles[0].accprof;
      callprofile_t callprof = this_stack.profiling.profiles[0].callprof;
      if (vftr_accprof_is_data_event (accprof.event_type)) {
         *tot_data_s += (double)callprof.time_excl_nsec / 1e9;
      } else if (vftr_accprof_is_compute_event (accprof.event_type)) {
         *tot_compute_s += (double)callprof.time_excl_nsec / 1e9;
      } else {
         *tot_wait_s += (double)callprof.time_excl_nsec / 1e9;
      }
   }
}

void vftr_write_ranklogfile_accprof_grouped_table (FILE *fp, stacktree_t stacktree, config_t config) {
   
   // Structure of OpenACC grouped table:
   // | SourceFile - line | Function | t_compute | t_data | t_wait | Bytes H -> D | Bytes D -> H | Bytes onDevice
   //
   // NOTE: Due to a bug in NVIDIA's OpenACC implementation, the source file might be inaccurate in some situations.
   //       This has especially been observed if OpenACC regions are implemented in header files.
   //       In that case, many more OpenACC calls are assigned to that header, although they are clearly not in there.
   //       Also, inlining might influence the accuracy of the line number assignment.

   uint64_t *region_ids = (uint64_t*)malloc (stacktree.nstacks * sizeof(uint64_t)); 
   char **region_names = (char**)malloc(stacktree.nstacks * sizeof(char*));
   char **func_names = (char**)malloc(stacktree.nstacks * sizeof(char*));
   for (int i = 0; i < stacktree.nstacks; i++) {
      region_ids[i] = 0;
   } 

   int n_region_ids = 0;
   for (int i = 0; i < stacktree.nstacks; i++) {
       accprofile_t accprof = stacktree.stacks[i].profiling.profiles[0].accprof;
       if (accprof.region_id != 0) {
          bool found = false;
          for (int j = 0; j < n_region_ids; j++) {
     	      if (region_ids[j] == accprof.region_id) {
                 found = true;
                 break;
              } 
          }
          if (!found) {
             region_ids[n_region_ids] = accprof.region_id;
             region_names[n_region_ids] = vftr_name_with_lines_1 (basename(accprof.source_file), accprof.line_start);
             func_names[n_region_ids] = accprof.func_name;
             n_region_ids++;
          }
       }
   } 

   double *t_summed_compute = (double*)malloc (n_region_ids * sizeof(double));
   double *t_summed_data = (double*)malloc (n_region_ids * sizeof(double));
   double *t_summed_wait = (double*)malloc (n_region_ids * sizeof(double));

   long long *bytes_summed_h2d = (long long*)malloc (n_region_ids * sizeof(long long));
   long long *bytes_summed_d2h = (long long*)malloc (n_region_ids * sizeof(long long));
   long long *bytes_summed_on_device = (long long*)malloc (n_region_ids * sizeof(long long));

   for (int i = 0; i < n_region_ids; i++) {
     t_summed_compute[i] = 0;
     t_summed_data[i] = 0;
     t_summed_wait[i] = 0;
     bytes_summed_h2d[i] = 0;
     bytes_summed_d2h[i] = 0;
     bytes_summed_on_device[i] = 0;
   }

   for (int i = 0; i < stacktree.nstacks; i++) {
      accprofile_t accprof = stacktree.stacks[i].profiling.profiles[0].accprof;
      if (accprof.region_id > 0) {
         int idx;
         for (idx = 0; idx < n_region_ids; idx++) {
            if (accprof.region_id == region_ids[idx]) break;
         }

         double this_t = (double)stacktree.stacks[i].profiling.profiles[0].callprof.time_excl_nsec / 1e9;
         if (vftr_accprof_is_compute_event (accprof.event_type)) {
            t_summed_compute[idx] += this_t;
         } else if (vftr_accprof_is_data_event (accprof.event_type)) {
            t_summed_data[idx] += this_t;
         } else {
            t_summed_wait[idx] += this_t;
         }

         if (vftr_accprof_is_h2d_event (accprof.event_type)) {
            bytes_summed_h2d[idx] += accprof.copied_bytes;
         } else if (vftr_accprof_is_d2h_event (accprof.event_type)) {
            bytes_summed_d2h[idx] += accprof.copied_bytes;
         } else if (vftr_accprof_is_ondevice_event (accprof.event_type)) {
            bytes_summed_on_device[idx] += accprof.copied_bytes;
         }
      }
   }
   
   double *t_summed_tot = (double*)malloc (n_region_ids * sizeof(double));
   long long *bytes_summed_tot = (long long*)malloc(n_region_ids * sizeof(long long));
   for (int i = 0; i < n_region_ids; i++) {
      t_summed_tot[i] = t_summed_compute[i] + t_summed_data[i] + t_summed_wait[i];
      bytes_summed_tot[i] = bytes_summed_h2d[i] + bytes_summed_d2h[i] + bytes_summed_on_device[i];
   }

   vftr_sort_arrays_for_grouped_table (config, n_region_ids,
				       t_summed_tot, bytes_summed_tot,
				       region_names, func_names,
				       t_summed_compute, t_summed_data, t_summed_wait,
				       bytes_summed_h2d, bytes_summed_d2h, bytes_summed_on_device);

   table_t table = vftr_new_table();
   vftr_table_set_nrows (&table, n_region_ids);
   
   vftr_table_add_column (&table, col_string, "Position", "%s", 'c', 'l', (void*)region_names);
   vftr_table_add_column (&table, col_string, "Function", "%s", 'c', 'r', (void*)func_names);
   vftr_table_add_column (&table, col_double, "t_compute [s]", "%.3lf", 'c', 'r', (void*)t_summed_compute);
   vftr_table_add_column (&table, col_double, "t_data [s]", "%.3lf", 'c', 'r', (void*)t_summed_data);
   vftr_table_add_column (&table, col_double, "t_wait [s]", "%.3lf", 'c', 'r', (void*)t_summed_wait);

   vftr_table_add_column (&table, col_longlong, "Host -> Device [B]", "%lld", 'c', 'r', (void*)bytes_summed_h2d);
   vftr_table_add_column (&table, col_longlong, "Device -> Host [B]", "%lld", 'c', 'r', (void*)bytes_summed_d2h);
   vftr_table_add_column (&table, col_longlong, "On Device [B]", "%lld", 'c', 'r', (void*)bytes_summed_on_device);

   fprintf (fp, "\n--OpenACC Summary--\n");
   fprintf (fp, "\n");
   vftr_print_accprof_gpuinfo (fp);
   fprintf (fp, "\n");
   vftr_print_table(fp, table);

   free (region_ids);
   free (region_names);
   free (func_names);
   free (t_summed_compute);
   free (t_summed_data);
   free (t_summed_wait);
   free (bytes_summed_h2d);
   free (bytes_summed_d2h);
   free (bytes_summed_on_device);
}


void vftr_write_ranklogfile_accprof_event_table (FILE *fp, stacktree_t stacktree, config_t config) {
   int n_stackids_with_accprof_data = 0;
   
   stack_t **sorted_stacks = vftr_sort_stacks_for_accprof (config, stacktree);
   
   for (int istack = 0; istack < stacktree.nstacks; istack++) {
      profile_t *this_profile = sorted_stacks[istack]->profiling.profiles;
      accprofile_t accprof = this_profile->accprof;
      if (accprof.event_type != 0) n_stackids_with_accprof_data++;
   }

   // Structure of OpenACC table:
   // | STID | event | #Calls | t_compute | t_data | t_wait | Bytes | Source File | Function
   //
   // The event is the acc_ev identifier, translated into a string.
   // For launch events, we add the kernel name in brackets behind that.
   //
   // Out of t_compute, t_data and t_wait, only one is non-zero for any acc_ev. We display them in separate columns anyway for better clarity.
   //
   // NOTE: Due to a bug in NVIDIA's OpenACC implementation, the source file might be inaccurate in some situations.
   //       This has especially been observed if OpenACC regions are implemented in header files.
   //       In that case, many more OpenACC calls are assigned to that header, although they are clearly not in there.
   //       Also, inlining might influence the accuracy of the line number assignment.


   int *stackids_with_accprof_data = (int*)malloc(n_stackids_with_accprof_data * sizeof(int));
   int *calls = (int*)malloc(n_stackids_with_accprof_data * sizeof(int));
   char **ev_names = (char**)malloc(n_stackids_with_accprof_data * sizeof(char*));
   size_t *copied_bytes = (size_t*)malloc(n_stackids_with_accprof_data * sizeof(size_t));
   double *t_compute = (double*)malloc(n_stackids_with_accprof_data * sizeof(double));
   double *t_data = (double*)malloc(n_stackids_with_accprof_data * sizeof(double));
   double *t_wait = (double*)malloc(n_stackids_with_accprof_data * sizeof(double));
   char **source_files = (char**)malloc(n_stackids_with_accprof_data * sizeof(char*));
   char **func_names = (char**)malloc(n_stackids_with_accprof_data * sizeof(char*));
   uint64_t *region_ids = (uint64_t*)malloc(n_stackids_with_accprof_data * sizeof(uint64_t));

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
      t_data[i] = 0.0;
      t_wait[i] = 0.0;
      if (vftr_accprof_is_compute_event (ev)) {
         t_compute[i] = t;
      } else if (vftr_accprof_is_data_event (ev)) {
         t_data[i] = t;
      } else {
         t_wait[i] = t;
      }
      // In rare cases, accprof does not return a source file. We are not yet sure why this happens and if this is a bug in the OpenACC implementation.
      if (accprof.source_file != NULL) {
         source_files[i] = vftr_name_with_lines_2 (basename(accprof.source_file), accprof.line_start, accprof.line_end);
      } else {
         source_files[i] = "unknown";
      }
      // We have not yet observed NULL function names, but better safe than sorry.
      func_names[i] = accprof.func_name != NULL ? accprof.func_name : "unknown";
      region_ids[i] = accprof.region_id;
      i++;
   }

   table_t table = vftr_new_table();
   vftr_table_set_nrows(&table, n_stackids_with_accprof_data);

   vftr_table_add_column (&table, col_int, "STID", "%d", 'c', 'r', (void*)stackids_with_accprof_data);
   vftr_table_add_column (&table, col_string, "event", "%s", 'c', 'r', (void*)ev_names);
   vftr_table_add_column (&table, col_longlong, "regionID", "0x%lx", 'c', 'l', (void*)region_ids);
   vftr_table_add_column (&table, col_int, "#Calls", "%d", 'c', 'r', (void*)calls);
   vftr_table_add_column (&table, col_double, "t_compute[s]", "%.3lf", 'c', 'r', (void*)t_compute);
   vftr_table_add_column (&table, col_double, "t_data[s]", "%.3lf", 'c', 'r', (void*)t_data);
   vftr_table_add_column (&table, col_double, "t_wait[s]", "%.3lf", 'c', 'r', (void*)t_wait);
   vftr_table_add_column (&table, col_long, "Bytes", "%ld", 'c', 'r', (void*)copied_bytes);
   vftr_table_add_column (&table, col_string, "Source File", "%s", 'c', 'l', (void*)source_files);
   vftr_table_add_column (&table, col_string, "Function", "%s", 'c', 'r', (void*)func_names);

   fprintf (fp, "\n--OpenACC Detailed Event Summary--\n");
   fprintf (fp, "\n");
   vftr_print_table(fp, table);

   free (stackids_with_accprof_data);
   free (ev_names);
   free (calls);
   free (t_compute);
   free (t_data);
   free (t_wait);
   free (copied_bytes);
   free (source_files);
   free (func_names);
   free (region_ids);
}
