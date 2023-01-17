#include <stdio.h>

#include "vftrace_state.h"
#include "configuration.h"

#include "hwprof_ve.h"

// This test simply checks that all VE counters can be retrieved and that
// their values are non-negative.

int main (int argc, char *argv[]) {
  config_t config = vftr_read_config();
  vftrace.hwprof_state.n_counters = config.hwprof.counters.hwc_name.n_elements;
  vftrace.hwprof_state.counters = (vftr_counter_t*)malloc(vftrace.hwprof_state.n_counters * sizeof(vftr_counter_t));
  for (int i = 0; i < vftrace.hwprof_state.n_counters; i++) {
    vftrace.hwprof_state.counters[i].name = config.hwprof.counters.hwc_name.values[i];
  }
  vftr_veprof_init();

  int n_active = 0;
  for (int i = 0; i < VE_MAX_HWC_EVENTS; i++) {
    if (vftrace.hwprof_state.veprof.active_counters[i]) {
       printf ("%d: %s\n", i, vftrace.hwprof_state.veprof.ve_hwc_names[i]);
       n_active++;
    }
  }
  printf ("active: %d\n", n_active);

  long long *counters = vftr_get_active_ve_counters ();

  int retval = 0;
  for (int i = 0; i < n_active; i++) {
     retval |= counters[i] < 0;
  }
  free(counters);
  printf ("Okay: %d\n", retval);
  return 0;
}
