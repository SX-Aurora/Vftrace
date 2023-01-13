#include "vftrace_state.h"

#define DUMMY_COUNTER_BASE 10

static long long n_dummys = 0;

long long *vftr_get_dummy_counters () {
   int n = vftrace.hwprof_state.n_counters;
   long long *counters = (long long *)malloc (n * sizeof(long long));
   for (int i = 0; i < n; i++) {
      counters[i] = n_dummys % DUMMY_COUNTER_BASE;
   }
   n_dummys++;
   return counters;
}
