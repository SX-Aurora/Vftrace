#include <string.h>
#include <papi.h>

#include "vftrace_state.h"

#include "papiprofiling_types.h"
#include "papi_calculator.h"

papiprofile_t vftr_new_papiprofiling () {
   papiprofile_t prof;
   int n_events = PAPI_num_events (vftrace.papi_state.eventset);
   prof.counters = (long long*)malloc (n_events * sizeof(long long));
   memset (prof.counters, 0, n_events * sizeof(long long));
   return prof;
}

void vftr_accumulate_papiprofiling (papiprofile_t *prof, bool is_entry) {
   int n = PAPI_num_events (vftrace.papi_state.eventset);
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

void vftr_papiprofiling_free (papiprofile_t *prof_ptr) {
   if (prof_ptr->counters != NULL) free (prof_ptr->counters);
   prof_ptr->counters = NULL;
}
