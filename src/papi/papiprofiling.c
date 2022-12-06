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

long long *vftr_get_papi_counters () {
  int n = PAPI_num_events (vftrace.papi_state.eventset);
  long long *counters = (long long *)malloc (n * sizeof(long long));
  int retval = PAPI_read (vftrace.papi_state.eventset, counters);
  return counters;
}

void vftr_accumulate_papiprofiling (papiprofile_t *prof, long long *counters, bool invert_sign) {
   int n = PAPI_num_events (vftrace.papi_state.eventset);
   for (int i = 0; i < n; i++) {
      if (invert_sign) {
         prof->counters[i] -= counters[i]; 
      } else {
         prof->counters[i] += counters[i]; 
      }
   }
}

void vftr_papiprofiling_free (papiprofile_t *prof_ptr) {
   if (prof_ptr->counters != NULL) free (prof_ptr->counters);
   prof_ptr->counters = NULL;
}
