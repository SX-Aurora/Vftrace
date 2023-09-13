#include <stdio.h>
#include <stdbool.h>

#include "configuration_types.h"
#include "symbols.h"
#include "tables.h"
#include "collated_stack_types.h"
#include "vftrace_state.h"
#include "sorting.h"
#include "misc_utils.h"

#include "cupti_vftr_callbacks.h"
#include "cudaprofiling_types.h"
#include "collated_cudaprofiling_types.h"
#include "cuda_utils.h"

void vftr_get_total_cuda_times_for_logfile (collated_stacktree_t stacktree, 
                                             float *tot_compute_s, float *tot_memcpy_s, float *tot_other_s) {
   *tot_compute_s = 0;
   *tot_memcpy_s = 0;
   *tot_other_s = 0;

   for (int istack = 0; istack < stacktree.nstacks; istack++) {
      collated_stack_t this_stack = stacktree.stacks[istack];
      collated_cudaprofile_t cudaprof = this_stack.profile.cudaprof;
      if (vftr_cuda_cbid_belongs_to_class (cudaprof.cbid, T_CUDA_COMP)) {
            *tot_compute_s += (float)cudaprof.t_ms / 1000;
      } else if (vftr_cuda_cbid_belongs_to_class (cudaprof.cbid, T_CUDA_MEMCP)) {
            *tot_memcpy_s += (float)cudaprof.t_ms / 1000;
      } else if (vftr_cuda_cbid_belongs_to_class (cudaprof.cbid, T_CUDA_OTHER)) {
            *tot_other_s += (float)cudaprof.t_ms / 1000;
      }
   }
}

typedef struct kernel_call_st {
  struct kernel_call_st *next;
  float avg_ncalls;
  int min_ncalls;
  int max_ncalls;
  int stack_id;
} kernel_call_t;

void vftr_extract_kernel_calls_all (collated_stack_t *stacks_ptr, int stack_id,
                                    kernel_call_t **kc_head, kernel_call_t **kc_current) {
   collated_stack_t stack = stacks_ptr[stack_id];
   if (stack.ncallees > 0) {
      for (int icallee = 0; icallee < stack.ncallees; icallee++) {
         vftr_extract_kernel_calls_all (stacks_ptr, stack.callees[icallee], kc_head, kc_current);
      }
   }
   collated_cudaprofile_t cudaprof = stack.profile.cudaprof;
   if (vftr_cuda_cbid_belongs_to_class (cudaprof.cbid, T_CUDA_COMP)) {
     if (*kc_head == NULL) {
        *kc_head = (kernel_call_t*)malloc(sizeof(kernel_call_t));
        *kc_current = *kc_head;
     } else {
        (*kc_current)->next = (kernel_call_t*)malloc(sizeof(kernel_call_t));
        *kc_current = (*kc_current)->next;
     }
     (*kc_current)->next = NULL;
     (*kc_current)->avg_ncalls = (float)cudaprof.avg_ncalls[0] / cudaprof.on_nranks;
     (*kc_current)->min_ncalls = cudaprof.min_ncalls[0];
     (*kc_current)->max_ncalls = cudaprof.max_ncalls[0];
     (*kc_current)->stack_id = stack_id;
   }
}

void vftr_write_cuda_memcpy_stats_all (FILE *fp, collated_stacktree_t stacktree) {

   int n_caller_max = 0;
   int n_avg_max = strlen("avg");
   int n_min_max = strlen("min");
   int n_max_max = strlen("max");

   for (int istack = 0; istack < stacktree.nstacks; istack++) {
      collated_stack_t this_stack = stacktree.stacks[istack];
      collated_cudaprofile_t cudaprof = this_stack.profile.cudaprof;
      int cbid = cudaprof.cbid;
      if (vftr_cuda_cbid_belongs_to_class (cbid, T_CUDA_MEMCP)) {
         int n = strlen(stacktree.stacks[this_stack.caller].name);
         if (n > n_caller_max) n_caller_max = n;
         int n1 = 3 + vftr_count_base_digits_float ((float)cudaprof.avg_ncalls[0] / cudaprof.on_nranks, 10);
         int n2 = 3 + vftr_count_base_digits_float ((float)cudaprof.avg_ncalls[1] / cudaprof.on_nranks, 10);
         if (n1 > n_avg_max) n_avg_max = n1;
         if (n2 > n_avg_max) n_avg_max = n2;
         n1 = vftr_count_base_digits (cudaprof.min_ncalls[0], 10);
         n2 = vftr_count_base_digits (cudaprof.min_ncalls[1], 10);
         if (n1 > n_min_max) n_min_max = n1; 
         if (n2 > n_min_max) n_min_max = n2; 
         n1 = vftr_count_base_digits (cudaprof.max_ncalls[0], 10);
         n2 = vftr_count_base_digits (cudaprof.max_ncalls[1], 10);
         if (n1 > n_max_max) n_max_max = n1;
         if (n2 > n_max_max) n_max_max = n2;
      }
   }

   int table_width = 18 + n_caller_max + n_avg_max + n_min_max + n_max_max;

   fprintf (fp, "\nCUDA ratio of memcpy / kernel calls: \n\n");
   for (int istack = 0; istack < stacktree.nstacks; istack++) {
      collated_stack_t this_stack = stacktree.stacks[istack];
      collated_cudaprofile_t cudaprof = this_stack.profile.cudaprof;
      int cbid = cudaprof.cbid;
      if (vftr_cuda_cbid_belongs_to_class (cbid, T_CUDA_MEMCP)) {
         for (int i = 0; i < table_width; i++) {
            fprintf (fp, "=");
         }
         fprintf (fp, "\n");
         fprintf (fp, "Memcpy caller: %*s %*s %*s %*s\n",
                      n_caller_max, stacktree.stacks[this_stack.caller].name,
                      n_avg_max, "avg",
                      n_min_max, "min",
                      n_max_max, "max");
         fprintf (fp, "           in: %*s %*.2f %*d %*d\n",
                      n_caller_max, "",
                      n_avg_max, (float)cudaprof.avg_ncalls[0] / cudaprof.on_nranks,
                      n_min_max, cudaprof.min_ncalls[0],
                      n_max_max, cudaprof.max_ncalls[0]);
         fprintf (fp, "          out: %*s %*.2f %*d %*d\n",
                      n_caller_max, "",
                      n_avg_max, (float)cudaprof.avg_ncalls[1] / cudaprof.on_nranks,
                      n_min_max, cudaprof.min_ncalls[1],
                      n_max_max, cudaprof.max_ncalls[1]);
         for (int i = 0; i < table_width; i++) {
            fprintf (fp, "=");
         }
         fprintf (fp, "\n");


         kernel_call_t *kc_head = NULL;
         kernel_call_t *kc_current = NULL;
         vftr_extract_kernel_calls_all (stacktree.stacks, this_stack.caller, &kc_head, &kc_current);
         kc_current = kc_head;

         int this_n_callee_max = strlen("Callee");
         int this_n_avg_max = strlen("avg");
         int this_n_min_max = strlen("min");
         int this_n_max_max = strlen("max");
         while (kc_current != NULL) {
            int n = strlen(stacktree.stacks[kc_current->stack_id].name);
            if (n > this_n_callee_max) this_n_callee_max = n;
            n = 3 + vftr_count_base_digits_float (kc_current->avg_ncalls, 10);
            if (n > this_n_avg_max) this_n_avg_max = n;
            n = vftr_count_base_digits (kc_current->min_ncalls, 10);
            if (n > this_n_min_max) this_n_min_max = n;
            n = vftr_count_base_digits (kc_current->max_ncalls, 10);
            if (n > this_n_max_max) this_n_max_max = n;
            kc_current = kc_current->next;
         }
         fprintf (fp, "%*s | %*s | %*s | %*s\n",
                  this_n_callee_max, "Callee",
                  this_n_avg_max, "avg",
                  this_n_min_max, "min",
                  this_n_max_max, "max");
  

         kc_current = kc_head;
         while (kc_current != NULL) {
           fprintf (fp, "%*s | %*.2f | %*d | %*d\n",
                    this_n_callee_max, stacktree.stacks[kc_current->stack_id].name,
                    this_n_avg_max, kc_current->avg_ncalls,
                    this_n_min_max, kc_current->min_ncalls,
                    this_n_max_max, kc_current->max_ncalls);
           kc_current =  kc_current->next;
         }
      }
   }
}

void vftr_write_logfile_cuda_table(FILE *fp, collated_stacktree_t stacktree, config_t config) {
   int n_stackids_with_cuda_data = 0;

   collated_stack_t **sorted_stacks = vftr_sort_collated_stacks_for_cuda (config, stacktree);

   for (int istack = 0; istack < stacktree.nstacks; istack++) {
      collated_cudaprofile_t cudaprof = sorted_stacks[istack]->profile.cudaprof;
      if (cudaprof.cbid != 0) n_stackids_with_cuda_data++;
   }

   // Structure of CUDA table:
   // | STID | cudaName | #Calls | CBID | t_compute | t_memcpy | t_other | memcpy_in | memcpy_out |
   //
   // The cudaName is in most cases  the symbol name of the calling CBID given by the callback interface.
   // I.e., library functions like "cudaMalloc" appear as "cudaMalloc".
   // CBIDs belonging to the LAUNCH group are assigned their function name. Hence, the name of the
   // CUDA kernel will be desplayed.
   //
   // Out of t_compute, t_memcpy and t_other, only one is non-zero for any CBID. We display them in separate
   // columns anyway for better clarity.

   int *stackids_with_cuda_data = (int*)malloc(n_stackids_with_cuda_data*sizeof(int));
   char **names = (char**)malloc(n_stackids_with_cuda_data*sizeof(char*));
   int *calls = (int*)malloc(n_stackids_with_cuda_data*sizeof(int));
   int *cbids = (int*)malloc(n_stackids_with_cuda_data*sizeof(int));
   float *t_compute = (float*)malloc(n_stackids_with_cuda_data*sizeof(float));
   float *t_memcpy = (float*)malloc(n_stackids_with_cuda_data*sizeof(float));
   float *t_other = (float*)malloc(n_stackids_with_cuda_data*sizeof(float));
   size_t *memcpy_in = (size_t*)malloc(n_stackids_with_cuda_data*sizeof(size_t));
   size_t *memcpy_out = (size_t*)malloc(n_stackids_with_cuda_data*sizeof(size_t));

   int i = 0;
   for (int istack = 0; istack < stacktree.nstacks; istack++) {
      collated_stack_t *this_stack = sorted_stacks[istack];
      collated_cudaprofile_t cudaprof = this_stack->profile.cudaprof;
      if (cudaprof.cbid == 0) continue;
      stackids_with_cuda_data[i] = istack;
      calls[i] = cudaprof.n_calls[0] + cudaprof.n_calls[1];
      cbids[i] = cudaprof.cbid;
      t_compute[i] = vftr_cuda_cbid_belongs_to_class (cbids[i], T_CUDA_COMP) ? cudaprof.t_ms / 1000 : 0;
      t_memcpy[i] = vftr_cuda_cbid_belongs_to_class (cbids[i], T_CUDA_MEMCP) ? cudaprof.t_ms / 1000 : 0;
      t_other[i] = vftr_cuda_cbid_belongs_to_class (cbids[i], T_CUDA_OTHER) ? cudaprof.t_ms / 1000 : 0;

      memcpy_in[i] = cudaprof.memcpy_bytes[CUDA_COPY_IN];
      memcpy_out[i] = cudaprof.memcpy_bytes[CUDA_COPY_OUT];
#ifdef _LIBERTY
      // Only makes a difference for kernel launches which are implemented in C++ and therefore
      // can have contrived mangled names.
      names[i] = vftr_demangle_cxx(this_stack->name);
#else
      names[i] = this_stack->name;
#endif
      i++;
   }

   table_t table = vftr_new_table();
   vftr_table_set_nrows(&table, n_stackids_with_cuda_data);

   vftr_table_add_column (&table, col_int, "STID", "%d", 'c', 'r', (void*)stackids_with_cuda_data);
   vftr_table_add_column (&table, col_string, "cudaName", "%s", 'c', 'r', (void*)names);
   vftr_table_add_column (&table, col_int, "CBID", "%d", 'c', 'r', (void*)cbids);
   vftr_table_add_column (&table, col_int, "#Calls", "%d", 'c', 'r', (void*)calls);
   vftr_table_add_column (&table, col_float, "t_compute[s]", "%.3f", 'c', 'r', (void*)t_compute);
   vftr_table_add_column (&table, col_float, "t_memcpy[s]", "%.3f", 'c', 'r', (void*)t_memcpy);
   vftr_table_add_column (&table, col_float, "t_other[s]", "%.3f", 'c', 'r', (void*)t_other);
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
   free(stackids_with_cuda_data);
}

void vftr_write_logfile_cbid_names (FILE *fp, collated_stacktree_t stacktree) {
   int n_different_cbids = 0;
   int *cbids_found = (int*)malloc(stacktree.nstacks * sizeof(int));
   const char **cbid_names = (const char**)malloc(stacktree.nstacks * sizeof(char*));
   for (int istack = 0; istack < stacktree.nstacks; istack++) {
      collated_stack_t this_stack = stacktree.stacks[istack];
      collated_cudaprofile_t cudaprof = this_stack.profile.cudaprof;
      if (cudaprof.cbid == 0) continue;
      bool cbid_present = false;
      for (int icbid = 0; icbid < n_different_cbids; icbid++) {
          if (cbids_found[icbid] == cudaprof.cbid) {
             cbid_present = true;
             break;
          }
      }
      if (!cbid_present) {
         cbids_found[n_different_cbids] = cudaprof.cbid;
         cuptiGetCallbackName (CUPTI_CB_DOMAIN_RUNTIME_API, cudaprof.cbid,
                               cbid_names + n_different_cbids);
         n_different_cbids++;
      }
   }

   table_t table = vftr_new_table();
   vftr_table_set_nrows (&table, n_different_cbids);
   vftr_table_left_outline (&table, false);
   vftr_table_right_outline (&table, false);
   vftr_table_columns_separating_line (&table, false);
  
   vftr_table_add_column(&table, col_int, "CBID", "%d", 'c', 'r', (void*)cbids_found); 
   vftr_table_add_column(&table, col_string, "Name", "%s", 'c', 'l', (void*)cbid_names);

   fprintf (fp, "\nCUPTI CBID names: \n");
   vftr_print_table(fp, table);
   vftr_table_free(&table);

   free(cbids_found);
   free(cbid_names);
}
