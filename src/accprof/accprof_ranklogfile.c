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
      vftr_stack_t this_stack = stacktree.stacks[istack];
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


   if (n_region_ids == 0) {
      fprintf (fp, "ACCProf: No OpenACC regions have been registered.\n");
      return;
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

typedef struct kernel_call_st {
  struct kernel_call_st *next;
  int n_calls;
  int stack_id;
} kernel_call_t;

int vftr_find_nonacc_root (vftr_stack_t *stacks, int stack_id) {
   accprofile_t accprof = stacks[stack_id].profiling.profiles[0].accprof;
   // This stack entry has an accprof entry. Go back one step.
   if (vftr_accprof_event_is_defined (accprof.event_type)) {
      return vftr_find_nonacc_root (stacks, stacks[stack_id].caller);
   } else {
      return stack_id;
   }
}

void vftr_extract_kernel_calls_acc (vftr_stack_t *stacks_ptr, int stack_id,
                                    kernel_call_t **kc_head, kernel_call_t **kc_current) {
  vftr_stack_t stack = stacks_ptr[stack_id];
  if (stack.ncallees > 0) {
     for (int icallee = 0; icallee < stack.ncallees; icallee++) {
        vftr_extract_kernel_calls_acc (stacks_ptr, stack.callees[icallee], kc_head, kc_current);
     }
  }
  accprofile_t accprof = stack.profiling.profiles[0].accprof;
  callprofile_t callprof = stack.profiling.profiles[0].callprof;
  if (vftr_accprof_is_launch_event (accprof.event_type)) {
     if (*kc_head == NULL) {
        *kc_head = (kernel_call_t*)malloc(sizeof(kernel_call_t));
        *kc_current = *kc_head;
     } else {
        (*kc_current)->next = (kernel_call_t*)malloc(sizeof(kernel_call_t));
        *kc_current = (*kc_current)->next;
     }
     (*kc_current)->next = NULL;
     (*kc_current)->n_calls = callprof.calls;
     (*kc_current)->stack_id = stack_id;
  } 
}

void vftr_write_accprof_memcpy_stats (FILE *fp, stacktree_t stacktree) {
   int *root_ids = (int*)malloc(stacktree.nstacks * sizeof(int));
   int *download_ids = (int*)malloc(stacktree.nstacks * sizeof(int));
   int *upload_ids = (int*)malloc(stacktree.nstacks * sizeof(int));

   for (int istack = 0; istack < stacktree.nstacks; istack++) {
      root_ids[istack] = 0;
      upload_ids[istack] = 0;
      download_ids[istack] = 0;
   }

   for (int istack = 0; istack < stacktree.nstacks; istack++) {
      vftr_stack_t this_stack = stacktree.stacks[istack];
      accprofile_t accprof = this_stack.profiling.profiles[0].accprof;
      if (accprof.event_type == acc_ev_enqueue_upload_start ||
          accprof.event_type == acc_ev_enqueue_download_start) {
         int root_id = vftr_find_nonacc_root (stacktree.stacks, istack);
         root_ids[root_id]++;
         if (accprof.event_type == acc_ev_enqueue_upload_start) {
            upload_ids[root_id] = istack;
         } else {
            download_ids[root_id] = istack;
         }
      }
   }

   for (int istack = 0; istack < stacktree.nstacks; istack++) {
      if (root_ids[istack] != 0 && root_ids[istack] != 2) {
         fprintf (fp, "Internal Vftrace error: \n");
         fprintf (fp, "OpenACC memcpy stats: root_id[%d] = %d\n", istack, root_ids[istack]);
         return;
      }
   }

   fprintf (fp, "\nOpenACC ratio of memcpy / kernel calls: \n\n");
   int table_width = 0;
   for (int istack = 0; istack < stacktree.nstacks; istack++) {
      if (root_ids[istack] > 0) {
         int root_id = istack;
         char *memcpy_caller_name = stacktree.stacks[root_id].name;
         int stack_id_upload = upload_ids[root_id];
         int n_upload = stack_id_upload > 0 ? stacktree.stacks[stack_id_upload].profiling.profiles[0].callprof.calls : 0;
         int stack_id_download = download_ids[root_id];
         int n_download = stack_id_download > 0 ? stacktree.stacks[stack_id_download].profiling.profiles[0].callprof.calls : 0;
         int t = 28 + strlen (memcpy_caller_name)
               + vftr_count_base_digits (n_upload, 10) + vftr_count_base_digits (n_download, 10);
         if (t > table_width) table_width = t;
      }
   }

   for (int istack = 0; istack < stacktree.nstacks; istack++) {
      if (root_ids[istack] > 0) {
         int root_id = istack;
         char *memcpy_caller_name = stacktree.stacks[root_id].name;
         int stack_id_upload = upload_ids[root_id];
         int n_upload = stack_id_upload > 0 ? stacktree.stacks[stack_id_upload].profiling.profiles[0].callprof.calls : 0;
         int stack_id_download = download_ids[root_id];
         int n_download = stack_id_download > 0 ? stacktree.stacks[stack_id_download].profiling.profiles[0].callprof.calls : 0;
         for (int i = 0; i < table_width; i++) {
            fprintf (fp, "=");
         }
         fprintf (fp, "\n");
         fprintf (fp, "Memcpy caller: %s, in: %d, out: %d\n", memcpy_caller_name,
                      n_upload, n_download);
         for (int i = 0; i < table_width; i++) {
            fprintf (fp, "=");
         }
         fprintf (fp, "\n");
         kernel_call_t *kc_head = NULL;
         kernel_call_t *kc_current = NULL;
         vftr_extract_kernel_calls_acc (stacktree.stacks, root_id, &kc_head, &kc_current);
         kc_current = kc_head;
         int max_n_callee = strlen("Callee");
         int max_n_calls = strlen ("n_calls");
         while (kc_current != NULL) {
            int n = strlen(stacktree.stacks[kc_current->stack_id].name);
            if (n > max_n_callee) max_n_callee = n;
            n = vftr_count_base_digits (kc_current->n_calls);
            if (n > max_n_calls) max_n_calls = n;
            kc_current = kc_current->next;
         }
         fprintf (fp, "%*s | %*s\n", max_n_callee, "Callee", max_n_calls, "n_calls");
         kc_current = kc_head;
         while (kc_current != NULL) {
            fprintf (fp, "%*s | %*d\n",
                     max_n_callee, stacktree.stacks[kc_current->stack_id].name,
                     max_n_calls, kc_current->n_calls);
            kc_current = kc_current->next;
         }
      }
   }
   
   free (root_ids);
   free (download_ids);
   free (upload_ids);
}

void vftr_write_ranklogfile_accprof_event_table (FILE *fp, stacktree_t stacktree, config_t config) {
   int n_stackids_with_accprof_data = 0;
   
   vftr_stack_t **sorted_stacks = vftr_sort_stacks_for_accprof (config, stacktree);
   
   for (int istack = 0; istack < stacktree.nstacks; istack++) {
      profile_t *this_profile = sorted_stacks[istack]->profiling.profiles;
      accprofile_t accprof = this_profile->accprof;
      if (accprof.event_type != 0) n_stackids_with_accprof_data++;
   }

   if (n_stackids_with_accprof_data == 0) {
      fprintf (fp, "ACCProf: No stacks with OpenACC events found\n");
      return;
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
      vftr_stack_t *this_stack = sorted_stacks[istack];
      profile_t *this_profile = this_stack->profiling.profiles;
      accprofile_t accprof = this_profile->accprof;
      callprofile_t callprof = this_profile->callprof;
      acc_event_t ev = accprof.event_type;
      if (ev == 0) continue;
      stackids_with_accprof_data[i] = this_stack->lid;
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
