#include <stdio.h>
#include <string.h>

#include "vftrace_state.h"
#include "configuration_types.h"
#include "symbols.h"
#include "tables.h"
#include "stack_types.h"
#include "sorting.h"
#include "collate_stacks.h"

#include "papiprofiling_types.h"
#include "papi_calculator.h"

void vftr_write_ranklogfile_papi_obs_table (FILE *fp, stacktree_t stacktree, config_t config) {
   vftr_stack_t **sorted_stacks = vftr_sort_stacks_for_prof (config, stacktree);

   fprintf (fp, "\nRuntime PAPI profile - Observables\n\n");

   int n_observables = vftrace.papi_state.calculator.n_observables;

   int n_without_init = 0;
   for (int istack = 0; istack < stacktree.nstacks; istack++) {
      vftr_stack_t *this_stack = sorted_stacks[istack];
      if (this_stack->lid != 0) n_without_init++;
   }
   int *calls = (int*)malloc(n_without_init * sizeof(int));
   char **func = (char**)malloc(n_without_init * sizeof(char*));
   double **observables = (double**)malloc(n_observables * sizeof(double*));
   for (int i = 0; i < n_observables; i++) {
      observables[i] = (double*)malloc(n_without_init * sizeof(double)); 
   }

   int idx = 0;
   for (int istack = 0; istack < stacktree.nstacks; istack++) {
      vftr_stack_t *this_stack = sorted_stacks[istack];
      if (this_stack->lid == 0) continue;
      callprofile_t callprof = this_stack->profiling.profiles[0].callprof;
      papiprofile_t papiprof = this_stack->profiling.profiles[0].papiprof;

      calls[idx] = callprof.calls;
      func[idx] = this_stack->name;

      vftr_set_papi_calculator_counters (&(vftrace.papi_state.calculator), papiprof.counters_excl);
      vftr_set_papi_calculator_builtin (&(vftrace.papi_state.calculator),
                                            PCALC_T, (double)callprof.time_excl_nsec * 1e-9);
      vftr_set_papi_calculator_builtin (&(vftrace.papi_state.calculator),
                                            PCALC_ONE, 1.0);

      for (int i = 0; i < n_observables; i++) {
         observables[i][idx] = vftr_papi_calculator_evaluate (vftrace.papi_state.calculator, i);   
      }
      idx++;
   }

   table_t table = vftr_new_table();
   vftr_table_set_nrows (&table, n_without_init); 

   vftr_table_add_column (&table, col_int, "#Calls", "%d", 'c', 'r', (void*)calls);
   vftr_table_add_column (&table, col_string, "Func", "%s", 'c', 'r', (void*)func);
   for (int i = 0; i < n_observables; i++) {
      char *obs_name = config.papi.observables.obs_name.values[i];
      char *obs_unit = config.papi.observables.unit.values[i];
      int slen = strlen(obs_name) + strlen(obs_unit) + 4;
      char *header = (char*)malloc(slen * sizeof(char));
      snprintf (header, slen, "%s [%s]", obs_name, obs_unit);
      vftr_table_add_column(&table, col_double, header, "%lf", 'c', 'r', (void*)observables[i]);
   }

   vftr_print_table (fp, table);
 
   vftr_table_free(&table);

   free (calls);
   free (func);
   free (observables);
}

void vftr_write_ranklogfile_papi_counter_table (FILE *fp, stacktree_t stacktree, config_t config) {
   
   vftr_stack_t **sorted_stacks = vftr_sort_stacks_for_prof (config, stacktree);

   fprintf (fp, "\nRuntime PAPI profile - Hardware Counter\n\n");

   int n_without_init = 0;
   for (int istack = 0; istack < stacktree.nstacks; istack++) {
      vftr_stack_t *this_stack = sorted_stacks[istack];
      if (this_stack->lid != 0) n_without_init++;
   }

   int *calls = (int*)malloc(n_without_init * sizeof(int));
   char **func = (char**)malloc(n_without_init * sizeof(char*));
   int n_counters = vftrace.papi_state.n_counters;
   long long **counters = (long long**)malloc(n_counters * sizeof(long long*));
   for (int i = 0; i < n_counters; i++) {
      counters[i] = (long long*)malloc(n_without_init * sizeof(long long));
      memset(counters[i], 0, n_without_init * sizeof(long long));
   }

   int idx = 0;
   for (int istack = 0; istack < stacktree.nstacks; istack++) {
      vftr_stack_t *this_stack = sorted_stacks[istack];
      if (this_stack->lid == 0) continue;

      callprofile_t callprof = this_stack->profiling.profiles[0].callprof;
      papiprofile_t papiprof = this_stack->profiling.profiles[0].papiprof;

      calls[idx] = callprof.calls;
      func[idx] = this_stack->name;

      for (int i = 0; i < n_counters; i++) {
         counters[i][idx] = papiprof.counters_excl[i];
      }
      idx++;
   }

   table_t table = vftr_new_table();
   vftr_table_set_nrows (&table, n_without_init);

   vftr_table_add_column (&table, col_int, "#Calls", "%d", 'c', 'r', (void*)calls);
   vftr_table_add_column (&table, col_string, "Func", "%s", 'c', 'r', (void*)func);
   for (int i = 0; i < n_counters; i++) {
      vftr_table_add_column(&table, col_longlong, vftrace.papi_state.counters[i].name,
                            "%lld", 'c', 'r', (void*)counters[i]);
   }

   vftr_print_table (fp, table);
   vftr_table_free (&table);

   free(calls);
   free(func);
   for (int i = 0; i < n_counters; i++) {
      free(counters[i]);
   }
   free(counters);
}
