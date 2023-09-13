#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <libgen.h>

#include "configuration_types.h"
#include "collated_stack_types.h"
#include "collated_profiling_types.h"
#include "callprofiling_types.h"
#include "tables.h"
#include "misc_utils.h"
#include "sorting.h"
#include "vftrace_state.h"

#include "accprof_init_final.h"
#include "accprofiling_types.h"
#include "accprof_events.h"

void vftr_get_total_accprof_times_for_logfile (collated_stacktree_t stacktree,
					       double *tot_compute_s, double *tot_data_s, double *tot_wait_s) {
   *tot_compute_s = 0;
   *tot_data_s = 0;
   *tot_wait_s = 0;

   for (int istack = 0; istack < stacktree.nstacks; istack++) {
      collated_stack_t this_stack = stacktree.stacks[istack];
      collated_callprofile_t callprof = this_stack.profile.callprof;
      collated_accprofile_t accprof = this_stack.profile.accprof;
      if (vftr_accprof_is_data_event (accprof.event_type)) {
         *tot_data_s += (double)callprof.time_excl_nsec / 1e9;
      } else if (vftr_accprof_is_compute_event (accprof.event_type)) {
         *tot_compute_s += (double)callprof.time_excl_nsec / 1e9;
      } else {
         *tot_wait_s += (double)callprof.time_excl_nsec / 1e9;
      }
   }
}


// Entry for the source file column, in the form "filename - line".
char *vftr_name_with_lines_1 (char *name, int line_1) {
   int n1 = strlen(name);
   int n2 = vftr_count_base_digits ((long long)line_1, 10);
   int slen = n1 + n2 + 4;
   char *s = (char*) malloc (slen * sizeof(char));
   snprintf (s, slen, "%s - %d", name, line_1); 
   return s;
}

// Entry for the source file column, in the form "filename (line1 - line2)".
char *vftr_name_with_lines_2 (char *name, int line_1, int line_2) {
   int n1 = strlen(name);
   int n2 = vftr_count_base_digits ((long long)line_1, 10);
   int n3 = vftr_count_base_digits ((long long)line_2, 10);
   int slen = n1 + n2 + n3 + 7;
   char *s = (char*) malloc (slen * sizeof(char));
   snprintf (s, slen, "%s (%d - %d)", name, line_1, line_2); 
   return s;
}

void vftr_sort_arrays_for_grouped_table (config_t config, int n_region_ids,
                                         double *t_tot, long long *bytes_tot,
				         char **region_names, char **func_names,
				         double *t_compute, double *t_data, double *t_wait,
				         long long *bytes_h2d, long long *bytes_d2h, long long *bytes_on_device) {

   char *column = config.accprof.sort_table.column.value;
   bool ascending = config.accprof.sort_table.ascending.value;

   int *perm = NULL;
   if (!strcmp(column, "time")) {
      vftr_sort_perm_double (n_region_ids, t_tot, &perm, ascending);
   } else if (!strcmp(column, "memcpy")) {
      vftr_sort_perm_longlong (n_region_ids, bytes_tot, &perm, ascending);
   } else {
      return;
   }

   vftr_apply_perm_charptr (n_region_ids, region_names, perm);
   vftr_apply_perm_charptr (n_region_ids, func_names, perm);
   vftr_apply_perm_double (n_region_ids, t_compute, perm);
   vftr_apply_perm_double (n_region_ids, t_data, perm);
   vftr_apply_perm_double (n_region_ids, t_wait, perm);
   vftr_apply_perm_longlong (n_region_ids, bytes_h2d, perm);
   vftr_apply_perm_longlong (n_region_ids, bytes_d2h, perm);
   vftr_apply_perm_longlong (n_region_ids, bytes_on_device, perm);
   free(perm);
}

bool vftr_has_accprof_events (collated_stacktree_t stacktree) {
   for (int i = 0; i < stacktree.nstacks; i++) {
      collated_accprofile_t accprof = stacktree.stacks[i].profile.accprof;
      if (accprof.region_id != 0) return true;
   }
   return false;
}

typedef struct kernel_call_st {
  struct kernel_call_st *next;
  float avg_ncalls;
  int min_ncalls;
  int max_ncalls;
  int stack_id;
} kernel_call_t;

void vftr_extract_kernel_calls_acc_all (collated_stack_t *stacks_ptr, int stack_id,
                                    kernel_call_t **kc_head, kernel_call_t **kc_current) {
  collated_stack_t stack = stacks_ptr[stack_id];
  if (stack.ncallees > 0) {
     for (int icallee = 0; icallee < stack.ncallees; icallee++) {
        vftr_extract_kernel_calls_acc_all (stacks_ptr, stack.callees[icallee], kc_head, kc_current);
     }
  }
  collated_accprofile_t accprof = stack.profile.accprof;
  collated_callprofile_t callprof = stack.profile.callprof;
  if (vftr_accprof_is_launch_event (accprof.event_type)) {
     if (*kc_head == NULL) {
        *kc_head = (kernel_call_t*)malloc(sizeof(kernel_call_t));
        *kc_current = *kc_head;
     } else {
        (*kc_current)->next = (kernel_call_t*)malloc(sizeof(kernel_call_t));
        *kc_current = (*kc_current)->next;
     }
     (*kc_current)->next = NULL;
     (*kc_current)->avg_ncalls = (float)accprof.ncalls[0] / accprof.on_nranks;
     (*kc_current)->min_ncalls = accprof.min_ncalls[0];
     (*kc_current)->max_ncalls = accprof.max_ncalls[0];
     (*kc_current)->stack_id = stack_id;
  } 
}

int vftr_find_nonacc_root_collated (collated_stack_t *stacks, int stack_id) {
   collated_accprofile_t accprof = stacks[stack_id].profile.accprof;
   // This stack entry has an accprof entry. Go back one step.
   if (vftr_accprof_event_is_defined (accprof.event_type)) {
      return vftr_find_nonacc_root_collated (stacks, stacks[stack_id].caller);
   } else {
      return stack_id;
   }
}


void vftr_write_accprof_memcpy_stats_all (FILE *fp, collated_stacktree_t stacktree) {
   int *root_ids = (int*)malloc(stacktree.nstacks * sizeof(int));
   int *download_ids = (int*)malloc(stacktree.nstacks * sizeof(int));
   int *upload_ids = (int*)malloc(stacktree.nstacks * sizeof(int));

   for (int istack = 0; istack < stacktree.nstacks; istack++) {
      root_ids[istack] = 0;
      upload_ids[istack] = 0;
      download_ids[istack] = 0;
   }

   for (int istack = 0; istack < stacktree.nstacks; istack++) {
      collated_stack_t this_stack = stacktree.stacks[istack];
      collated_accprofile_t accprof = this_stack.profile.accprof;
      if (accprof.event_type == acc_ev_enqueue_upload_start ||
          accprof.event_type == acc_ev_enqueue_download_start) {
         int root_id = vftr_find_nonacc_root_collated (stacktree.stacks, istack);
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

   fprintf (fp, "\nOpenACC ratio of memcpy / kernel calls: \n");
   for (int istack = 0; istack < stacktree.nstacks; istack++) {
      if (root_ids[istack] > 0) {
         int root_id = istack;
         int stack_id_upload = upload_ids[root_id];
         int n_upload = stack_id_upload > 0 ? stacktree.stacks[stack_id_upload].profile.callprof.calls : 0;
         int stack_id_download = download_ids[root_id];
         int n_download = stack_id_download > 0 ? stacktree.stacks[stack_id_download].profile.callprof.calls : 0;
         collated_accprofile_t accprof_up = stacktree.stacks[stack_id_upload].profile.accprof;
         collated_accprofile_t accprof_down = stacktree.stacks[stack_id_download].profile.accprof;
         fprintf (fp, "%s:    in: %.2f %d %d, out: %.2f %d %d\n",
                      stacktree.stacks[root_id].name,
                      (float)accprof_up.avg_ncalls[0] / accprof_up.on_nranks,
                      accprof_up.max_ncalls[0], accprof_up.min_ncalls[0],
                      (float)accprof_down.avg_ncalls[1] / accprof_down.on_nranks,
                      accprof_down.max_ncalls[1], accprof_down.min_ncalls[1]);

         kernel_call_t *kc_head = NULL;
         kernel_call_t *kc_current = NULL;
         vftr_extract_kernel_calls_acc_all (stacktree.stacks, root_id, &kc_head, &kc_current);
         kc_current = kc_head;
         while (kc_current != NULL) {
            fprintf (fp, "  ->  %s:  %.2f %d %d\n",
                     stacktree.stacks[kc_current->stack_id].name,
                     kc_current->avg_ncalls,
                     kc_current->min_ncalls,
                     kc_current->max_ncalls);
            kc_current = kc_current->next;
         }
      }
   }
   
   free (root_ids);
   free (download_ids);
   free (upload_ids);

}

void vftr_write_logfile_accprof_grouped_table (FILE *fp, collated_stacktree_t stacktree, config_t config) {
   
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
       collated_accprofile_t accprof = stacktree.stacks[i].profile.accprof;
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
      collated_accprofile_t accprof = stacktree.stacks[i].profile.accprof;
      if (accprof.region_id > 0) {
         int idx;
         for (idx = 0; idx < n_region_ids; idx++) {
            if (accprof.region_id == region_ids[idx]) break;
         }

         double this_t = (double)stacktree.stacks[i].profile.callprof.time_excl_nsec / 1e9;
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

   free (t_summed_tot);
   free (bytes_summed_tot);
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

void vftr_write_logfile_accprof_event_table (FILE *fp, collated_stacktree_t stacktree, config_t config) {
   int n_stackids_with_accprof_data = 0;

   collated_stack_t **sorted_stacks = vftr_sort_collated_stacks_for_accprof (config, stacktree);

   for (int istack = 0; istack < stacktree.nstacks; istack++) {
      collated_accprofile_t accprof = sorted_stacks[istack]->profile.accprof;
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
      collated_stack_t *this_stack = sorted_stacks[istack];
      collated_accprofile_t accprof = this_stack->profile.accprof;
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
