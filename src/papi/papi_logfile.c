#include <stdio.h>
#include <string.h>

#include "vftrace_state.h"
#include "configuration_types.h"
#include "collated_stack_types.h"
#include "table_types.h"
#include "tables.h"
#include "sorting.h"
#include "collate_stacks.h"

#include "papiprofiling_types.h"

void vftr_write_papi_table (FILE *fp, collated_stacktree_t stacktree, config_t config) {

   collated_stack_t **sorted_stacks =
      vftr_sort_collated_stacks_for_prof(config, stacktree);

   fprintf (fp, "\nRuntime PAPI profile\n");

   int n_observables = vftrace.papi_state.calculator.n_observables;

   int n_without_init = 0;
   for (int istack = 0; istack < stacktree.nstacks; istack++) {
      collated_stack_t *this_stack = sorted_stacks[istack];
      if (!vftr_collstack_is_init(*this_stack)) n_without_init++;
   }
   int *calls = (int*)malloc(n_without_init * sizeof(int));
   char **func = (char**)malloc(n_without_init * sizeof(char*));
   double **observables = (double**)malloc(n_observables * sizeof(double*));
   for (int i = 0; i < n_observables; i++) {
      observables[i] = (double*)malloc(n_without_init * sizeof(double)); 
   }

   int idx = 0;
   for (int istack = 0; istack < stacktree.nstacks; istack++) {
      collated_stack_t *this_stack = sorted_stacks[istack];
      if (vftr_collstack_is_init(*this_stack)) continue;
      collated_callprofile_t callprof = this_stack->profile.callprof;
      papiprofile_t papiprof = this_stack->profile.papiprof;

      calls[idx] = callprof.calls;
      func[idx] = this_stack->name;

      for (int i = 0; i < n_observables; i++) {
         observables[i][idx] = papiprof.observables[i];
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

void vftr_write_papi_counter_summary (FILE *fp, collated_stacktree_t stacktree, config_t config) {
   int n_events = PAPI_num_events (vftrace.papi_state.eventset);
   if (n_events == 0) {
      fprintf (fp, "\nNo hardware counters registered.\n");
      return;
   }
   long long *counter_sum = (long long*)malloc(n_events * sizeof(long long));
   memset (counter_sum, 0, n_events * sizeof(long long));
   for (int istack = 0; istack < stacktree.nstacks; istack++) {
      collated_stack_t this_stack = stacktree.stacks[istack]; 
      papiprofile_t papiprof = this_stack.profile.papiprof;

      for (int e = 0; e < n_events; e++) {
         counter_sum[e] += papiprof.counters_excl[e]; 
      }
   }

   fprintf (fp, "Total PAPI counters: \n\n");
   for (int e = 0; e < vftrace.papi_state.n_counters; e++) {
      fprintf (fp, "  %s: %lld\n",  vftrace.papi_state.counters[e].name, counter_sum[e]);
   }
   free (counter_sum);
}

void vftr_write_event_descriptions (FILE *fp) {
   //for (int i = 0; i < vftrace.papi_state.n_available_events; i++) {
   //   fprintf (fp, "%s: %s\n", vftrace.papi_state.event_names[i], vftrace.papi_state.event_descriptions[i]);
   //}
}
