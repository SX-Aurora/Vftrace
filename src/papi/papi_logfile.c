#include <stdio.h>

#include "vftrace_state.h"
#include "configuration_types.h"
#include "collated_stack_types.h"

#include "papiprofiling_types.h"

void vftr_write_papi_table (FILE *fp, collated_stacktree_t stacktree, config_t config) {
   int n_events = vftrace.papi_state.n_available_events;
   long long **counters = (long long**)malloc (n_events * sizeof(long long*));

   for (int i = 0; i < stacktree.nstacks; i++) {
      papiprofile_t papiprof = stacktree.stacks[i].profile.papiprof;
      //callprofile_t callprof = stacktree.stacks[i].profile.callprof;
      fprintf (fp, "StackID %d: \n", i);
      for (int e = 0; e < n_events; e++) {
         fprintf (fp, "    %s: %lld %s\n", vftrace.papi_state.event_names[e],
                                           papiprof.counters[e],
        				   vftrace.papi_state.event_units[e]);
      }
   }
   free(counters);
}

void vftr_write_event_descriptions (FILE *fp) {
   for (int i = 0; i < vftrace.papi_state.n_available_events; i++) {
      fprintf (fp, "%s: %s\n", vftrace.papi_state.event_names[i], vftrace.papi_state.event_descriptions[i]);
   }
}
