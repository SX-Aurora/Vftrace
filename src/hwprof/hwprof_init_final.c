#include <stdlib.h>

#include "vftrace_state.h"

#include "calculator.h"

void vftr_hwprof_init (config_t config) {
   vftrace.hwprof_state.n_counters = config.hwprof.counters.hwc_name.n_elements;
   vftrace.hwprof_state.counters = (vftr_counter_t*)malloc(vftrace.hwprof_state.n_counters * sizeof(vftr_counter_t));

   for (int i = 0; i < vftrace.hwprof_state.n_counters; i++) {
      vftrace.hwprof_state.counters[i].name = config.hwprof.counters.hwc_name.values[i];
   }

   vftrace.hwprof_state.n_observables = config.hwprof.observables.obs_name.n_elements;

   char **symbols = (char**)malloc(vftrace.hwprof_state.n_counters * sizeof(char*));
   for (int i = 0; i < vftrace.hwprof_state.n_counters; i++) {
      symbols[i] = config.hwprof.counters.symbol.values[i];
   }

   vftrace.hwprof_state.calculator = vftr_init_calculator (vftrace.hwprof_state.n_observables, symbols, 
                                        config.hwprof.observables.formula_expr.values);
   free(symbols);

}

void vftr_hwprof_finalize () {
   if (vftrace.hwprof_state.counters != NULL) free(vftrace.hwprof_state.counters);
   vftrace.hwprof_state.counters = NULL;
}
