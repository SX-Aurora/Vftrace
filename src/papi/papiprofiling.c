#include <string.h>

#include "vftrace_state.h"

#include "papiprofiling_types.h"

papiprofile_t vftr_new_papiprofiling () {
   papiprofile_t prof;
   prof.counters = (long long*)malloc (vftrace.papi_state.n_available_events * sizeof(long long));
   memset (prof.counters, 0, vftrace.papi_state.n_available_events * sizeof(long long));
   return prof;
}

void vftr_accumulate_papiprofiling (papiprofile_t *prof, bool is_entry) {
   int n = vftrace.papi_state.n_available_events;
   long long *counters = (long long*)malloc (n * sizeof(long long));
   int retval = PAPI_read (vftrace.papi_state.eventset, counters);

   for (int i = 0; i < n; i++) {
      if (is_entry) {
         prof->counters[i] -= counters[i]; 
      } else {
         prof->counters[i] += counters[i]; 
      }
   }
   free(counters);
}
