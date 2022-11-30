#include <stdio.h>

#include "vftrace_state.h"
#include "configuration_types.h"
#include "collated_stack_types.h"

#include "papiprofiling_types.h"
#include "papi_calculator.h"

void vftr_write_papi_table (FILE *fp, collated_stacktree_t stacktree, config_t config) {
   int n_events = PAPI_num_events (vftrace.papi_state.eventset);
   long long **counters = (long long**)malloc (n_events * sizeof(long long*));

   for (int i = 0; i < stacktree.nstacks; i++) {
      papiprofile_t papiprof = stacktree.stacks[i].profile.papiprof;
      collated_callprofile_t callprof = stacktree.stacks[i].profile.callprof;
      fprintf (fp, "StackID %d: \n", i);
      for (int e = 0; e < n_events; e++) {
         fprintf (fp, "    %s: %lld / %lld\n", config.papi.counters.native_name.values[e],
                                               papiprof.counters[e], callprof.time_excl_nsec);
      }
      //for (int e = 0; e < n_events; e++) {
      //   fprintf (fp, "    %s: %lld %s\n", vftrace.papi_state.event_names[e],
      //                                     papiprof.counters[e],
      //  				   vftrace.papi_state.event_units[e]);
      //}
      //
          
      vftr_set_papi_calculator_counters (&(vftrace.papi_state.calculator), papiprof.counters);
      vftr_set_papi_calculator_builtins (&(vftrace.papi_state.calculator), (double)callprof.time_excl_nsec * 1e-9);
      vftr_print_papi_calculator_state (vftrace.papi_state.calculator);

      for (int e = 0; e < n_events; e++) {
         printf ("Observable: %lf\n", vftr_papi_calculator_evaluate (vftrace.papi_state.calculator, e));
      }
   }
   free(counters);
}

void vftr_write_event_descriptions (FILE *fp) {
   for (int i = 0; i < vftrace.papi_state.n_available_events; i++) {
      fprintf (fp, "%s: %s\n", vftrace.papi_state.event_names[i], vftrace.papi_state.event_descriptions[i]);
   }
}
