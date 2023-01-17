#include "hwprof_ve.h"

// This test simply checks that all VE counters can be retrieved and that
// their values are non-negative.

int main (int argc, char *argv[]) {
  long long *counters = vftr_get_all_ve_counters ();

  int retval = 0;
  for (int i = 0; i < VE_MAX_HWC_EVENTS; i++) {
     printf ("%d: %lld\n", i, counters[i]);
     retval |= counters[i] < 0;
  }
  free(counters);
  return retval;
}
