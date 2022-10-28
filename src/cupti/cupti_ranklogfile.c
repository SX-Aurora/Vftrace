#include <stdio.h>

#include "configuration_types.h"
#include "symbols.h"
#include "tables.h"
#include "stack_types.h"
#include "sorting.h"

#include "cupti_vftr_callbacks.h"
#include "cuptiprofiling_types.h"
#include "cupti_utils.h"

void vftr_get_total_cupti_times_for_ranklogfile (stacktree_t stacktree, float *tot_compute_s,
                                                 float *tot_memcpy_s, float *tot_other_s) {
   *tot_compute_s = 0;
   *tot_memcpy_s = 0;
   *tot_other_s = 0;

   for (int istack = 0; istack < stacktree.nstacks; istack++) {
      stack_t this_stack = stacktree.stacks[istack];
      cuptiprofile_t cuptiprof = this_stack.profiling.profiles[0].cuptiprof;
      if (vftr_cupti_cbid_belongs_to_class (cuptiprof.cbid, T_CUPTI_COMP))  {
            *tot_compute_s += cuptiprof.t_ms / 1000;
      } else if (vftr_cupti_cbid_belongs_to_class (cuptiprof.cbid, T_CUPTI_MEMCP)) {
            *tot_memcpy_s += cuptiprof.t_ms / 1000;
      } else if (vftr_cupti_cbid_belongs_to_class (cuptiprof.cbid, T_CUPTI_OTHER)) {
            *tot_other_s += cuptiprof.t_ms / 1000;
      }
   }
}

void vftr_write_ranklogfile_cupti_table(FILE *fp, stacktree_t stacktree, config_t config) {
   int n_stackids_with_cupti_data = 0;

   stack_t **sorted_stacks = vftr_sort_stacks_for_cupti (config, stacktree);

   for (int istack = 0; istack < stacktree.nstacks; istack++) {
      // CUPTI only supported for one thread, thus there is only one profile.
      profile_t *this_profile = sorted_stacks[istack]->profiling.profiles;
      if (this_profile->cuptiprof.cbid != 0) n_stackids_with_cupti_data++;
   }

   // Structure of CUPTI table:
   // | STID | cudaName | #Calls | CBID | t_compute | t_memcpy | t_other | memcpy_in | memcpy_out |
   //
   // The cudaName is in most cases  the symbol name of the calling CBID given by the callback interface.
   // I.e., library functions like "cudaMalloc" appear as "cudaMalloc".
   // CBIDs belonging to the LAUNCH group are assigned their function name. Hence, the name of the
   // CUDA kernel will be desplayed.
   //
   // Out of t_compute, t_memcpy and t_other, only one is non-zero for any CBID. We display them in separate
   // columns anyway for better clarity.

   int *stackids_with_cupti_data = (int*)malloc(n_stackids_with_cupti_data*sizeof(int));
   char **names = (char**)malloc(n_stackids_with_cupti_data*sizeof(char*));
   int *calls = (int*)malloc(n_stackids_with_cupti_data*sizeof(int));
   int *cbids = (int*)malloc(n_stackids_with_cupti_data*sizeof(int));
   float *t_compute = (float*)malloc(n_stackids_with_cupti_data*sizeof(float));
   float *t_memcpy = (float*)malloc(n_stackids_with_cupti_data*sizeof(float));
   float *t_other = (float*)malloc(n_stackids_with_cupti_data*sizeof(float));
   size_t *memcpy_in = (size_t*)malloc(n_stackids_with_cupti_data*sizeof(size_t));
   size_t *memcpy_out = (size_t*)malloc(n_stackids_with_cupti_data*sizeof(size_t));

   int i = 0;
   for (int istack = 0; istack < stacktree.nstacks; istack++) {
      stack_t *this_stack = sorted_stacks[istack];
      profile_t *this_profile = this_stack->profiling.profiles;
      cuptiprofile_t cuptiprof = this_profile->cuptiprof;

      int cbid = cuptiprof.cbid;
      if (cbid == 0) continue;
      stackids_with_cupti_data[i] = istack;
      calls[i] = cuptiprof.n_calls;
      cbids[i] = cbid;
      t_compute[i] = vftr_cupti_cbid_belongs_to_class (cbids[i], T_CUPTI_COMP) ? cuptiprof.t_ms / 1000 : 0;
      t_memcpy[i] = vftr_cupti_cbid_belongs_to_class (cbids[i], T_CUPTI_MEMCP) ? cuptiprof.t_ms / 1000 : 0;
      t_other[i] = vftr_cupti_cbid_belongs_to_class (cbids[i], T_CUPTI_OTHER) ? cuptiprof.t_ms / 1000 : 0;
      memcpy_in[i] = cuptiprof.memcpy_bytes[CUPTI_COPY_IN];
      memcpy_out[i] = cuptiprof.memcpy_bytes[CUPTI_COPY_OUT];
#ifdef _LIBERTY
      // Only makes a difference for kernel launches which are implemented in C++ and therefore
      // can have contrived mangled names.
      names[i] = vftr_demangle_cxx(this_stack->name);
#else
      names[i] = this_stack.name;
#endif
      i++;
  }

  table_t table = vftr_new_table();
  vftr_table_set_nrows(&table, n_stackids_with_cupti_data);

  vftr_table_add_column (&table, col_int, "STID", "%d", 'c', 'r', (void*)stackids_with_cupti_data);
  vftr_table_add_column (&table, col_string, "cudaName", "%s", 'c', 'r', (void*)names);
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
  free(stackids_with_cupti_data);
}

void vftr_write_ranklogfile_cbid_names (FILE *fp, stacktree_t stacktree) {
   int n_different_cbids = 0;
   int *cbids_found = (int*)malloc(stacktree.nstacks * sizeof(int));
   const char **cbid_names = (const char**)malloc(stacktree.nstacks * sizeof(char*));
   for (int istack = 0; istack < stacktree.nstacks; istack++) {
      stack_t this_stack = stacktree.stacks[istack];
      cuptiprofile_t cuptiprof = this_stack.profiling.profiles[0].cuptiprof;
      if (cuptiprof.cbid == 0) continue;
      bool cbid_present = false;
      for (int icbid = 0; icbid < n_different_cbids; icbid++) {
          if (cbids_found[icbid] == cuptiprof.cbid) {
             cbid_present = true;
             break;
          }
      }
      if (!cbid_present) {
         cbids_found[n_different_cbids] = cuptiprof.cbid;
         cuptiGetCallbackName (CUPTI_CB_DOMAIN_RUNTIME_API, cuptiprof.cbid,
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
