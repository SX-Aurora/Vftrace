#include <stdio.h>
#include <string.h>

#include "vftrace_state.h"
#include "configuration_types.h"
#include "collated_stack_types.h"
#include "table_types.h"
#include "tables.h"
#include "sorting.h"
#include "collate_stacks.h"

#include "hwprof_state_types.h"
#include "hwprofiling_types.h"

char *vftr_hwtype_string (int hwtype) {
   switch (hwtype) {
      case HWC_NONE:
         return "None";
      case HWC_DUMMY:
         return "Dummy";
      case HWC_PAPI:
         return "PAPI";
      case HWC_LIKWID:
         return "Likwid";
      case HWC_VE:
         return "VEPERF";
      default:
         return "Unknown";
   }
}

void vftr_write_hwprof_observables_table (FILE *fp, collated_stacktree_t stacktree, config_t config) {

   collated_stack_t **sorted_stacks =
      vftr_sort_collated_stacks_hwprof_obs(config, stacktree);

   fprintf (fp, "\nHWProf - Observables (%s)\n\n",
            vftr_hwtype_string(vftrace.hwprof_state.hwc_type));

   int n_observables = vftrace.hwprof_state.n_observables;

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
      hwprofile_t hwprof = this_stack->profile.hwprof;

      calls[idx] = callprof.calls;
      func[idx] = this_stack->name;

      for (int i = 0; i < n_observables; i++) {
         observables[i][idx] = hwprof.observables[i];
      }
      idx++;
   }

   table_t table = vftr_new_table();
   vftr_table_set_nrows (&table, n_without_init); 

   vftr_table_add_column (&table, col_int, "#Calls", "%d", 'c', 'r', (void*)calls);
   vftr_table_add_column (&table, col_string, "Func", "%s", 'c', 'r', (void*)func);
   for (int i = 0; i < n_observables; i++) {
      char *obs_name = vftrace.hwprof_state.observables[i].name;
      char *obs_unit;
      if (vftrace.hwprof_state.observables[i].unit != NULL) {
          obs_unit = vftrace.hwprof_state.observables[i].unit;
      } else {
          obs_unit = "";
      }
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

void vftr_write_hwprof_observables_logfile_summary (FILE *fp, collated_stacktree_t stacktree) {
   int n_observables = vftrace.hwprof_state.n_observables;
   if (n_observables == 0) {
      fprintf (fp, "\nNo observables registered.\n");
      return;
   }

   double *obs_sum = (double*)malloc(n_observables * sizeof(double));
   memset (obs_sum, 0, n_observables * sizeof(double));
   for (int istack = 0; istack < stacktree.nstacks; istack++) {
      collated_stack_t this_stack = stacktree.stacks[istack];
      hwprofile_t hwprof = this_stack.profile.hwprof;
     
      for (int i = 0; i < n_observables; i++) {
         obs_sum[i] += hwprof.observables[i];
      } 
   } 

   fprintf (fp, "HWProf observables summary: \n\n");
   for (int i = 0; i < n_observables; i++) {
      if (vftrace.hwprof_state.observables[i].unit != NULL) {
         fprintf (fp, "  %s: %lf %s\n", vftrace.hwprof_state.observables[i].name,
                                        obs_sum[i], vftrace.hwprof_state.observables[i].unit);
      } else {
         fprintf (fp, "  %s: %lf\n", vftrace.hwprof_state.observables[i].name, obs_sum[i]);
      }
   }
   free(obs_sum);
}

void vftr_write_hwprof_counter_logfile_summary (FILE *fp, collated_stacktree_t stacktree) {
   int n_counters = vftrace.hwprof_state.n_counters;
   if (n_counters == 0) {
      fprintf (fp, "\nNo hardware counters registered.\n");
      return;
   }
   long long *counter_sum = (long long*)malloc(n_counters * sizeof(long long));
   memset (counter_sum, 0, n_counters * sizeof(long long));
   for (int istack = 0; istack < stacktree.nstacks; istack++) {
      collated_stack_t this_stack = stacktree.stacks[istack]; 
      hwprofile_t hwprof = this_stack.profile.hwprof;

      for (int e = 0; e < n_counters; e++) {
         counter_sum[e] += hwprof.counters_excl[e]; 
      }
   }

   fprintf (fp, "HWProf counters summary: \n\n");
   for (int e = 0; e < vftrace.hwprof_state.n_counters; e++) {
      fprintf (fp, "  %s: %lld\n",  vftrace.hwprof_state.counters[e].name, counter_sum[e]);
   }
   free (counter_sum);
}

void vftr_write_logfile_hwprof_counter_table (FILE *fp, collated_stacktree_t stacktree, config_t config) {
   collated_stack_t **sorted_stacks =
      vftr_sort_collated_stacks_hwprof_obs(config, stacktree);

   fprintf (fp, "\nHWProf - Hardware Counters (%s)\n\n",
            vftr_hwtype_string(vftrace.hwprof_state.hwc_type));

   int n_without_init = 0;
   for (int istack = 0; istack < stacktree.nstacks; istack++) {
      collated_stack_t *this_stack = sorted_stacks[istack];
      if (!vftr_collstack_is_init(*this_stack)) n_without_init++;
   }

   int *calls = (int*)malloc(n_without_init * sizeof(int));
   char **func = (char**)malloc(n_without_init * sizeof(char*));
   int n_counters = vftrace.hwprof_state.n_counters;
   long long **counters = (long long**)malloc(n_counters * sizeof(long long*));
   for (int i = 0; i < n_counters; i++) {
      counters[i] = (long long*)malloc(n_without_init * sizeof(long long)); 
      memset(counters[i], 0, n_without_init * sizeof(long long));
   }

   int idx = 0;
   for (int istack = 0; istack < stacktree.nstacks; istack++) {
      collated_stack_t *this_stack = sorted_stacks[istack];
      if (vftr_collstack_is_init(*this_stack)) continue;

      collated_callprofile_t callprof = this_stack->profile.callprof;
      hwprofile_t hwprof = this_stack->profile.hwprof;

      calls[idx] = callprof.calls;
      func[idx] = this_stack->name;

      for (int i = 0; i < n_counters; i++) {
         counters[i][idx] = hwprof.counters_excl[i];
      }
      idx++;
   }

   table_t table = vftr_new_table();
   vftr_table_set_nrows (&table, n_without_init); 

   vftr_table_add_column (&table, col_int, "#Calls", "%d", 'c', 'r', (void*)calls);
   vftr_table_add_column (&table, col_string, "Func", "%s", 'c', 'r', (void*)func);
   for (int i = 0; i < n_counters; i++) {
      vftr_table_add_column(&table, col_longlong, vftrace.hwprof_state.counters[i].name,
                            "%lld", 'c', 'r', (void*)counters[i]);
   }

   vftr_print_table (fp, table);
 
   vftr_table_free(&table);

   free (calls);
   free (func);
   free (counters);

}
