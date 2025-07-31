#include <stdlib.h>
#include <string.h>

#include "vftrace_state.h"

#include "hwprof_state_types.h"
#include "calculator.h"
#ifdef _PAPI_AVAIL
#include "hwprof_papi.h"
#endif
#include "hwprof_likwid.h"

char *vftr_builtin_obs_symbols[NSYM_BUILTIN] = {"T", "CALLS"};

void vftr_hwprof_init (config_t config) {
   // Invalid values should be caught in the assertion
   if (!strcmp(config.hwprof.hwc_type.value, "dummy")) {
      vftrace.hwprof_state.hwc_type = HWC_DUMMY;
   } else if (!strcmp(config.hwprof.hwc_type.value, "papi")) {
      vftrace.hwprof_state.hwc_type = HWC_PAPI;
   } else if (!strcmp(config.hwprof.hwc_type.value, "likwid")) {
      vftrace.hwprof_state.hwc_type = HWC_LIKWID;
   } else if (!strcmp(config.hwprof.hwc_type.value, "ve")) {
      vftrace.hwprof_state.hwc_type = HWC_VE;
   }

   vftrace.hwprof_state.active = config.hwprof.active.value;

   vftrace.hwprof_state.n_counters = config.hwprof.counters.hwc_name.n_elements;
   vftrace.hwprof_state.counters = (vftr_counter_t*)malloc(vftrace.hwprof_state.n_counters * sizeof(vftr_counter_t));

   for (int i = 0; i < vftrace.hwprof_state.n_counters; i++) {
      vftrace.hwprof_state.counters[i].name = config.hwprof.counters.hwc_name.values[i];
      if (config.hwprof.counters.symbol.set) {
         vftrace.hwprof_state.counters[i].symbol = config.hwprof.counters.symbol.values[i];
      } else {
         vftrace.hwprof_state.counters[i].symbol = NULL;
      }
   }

   vftrace.hwprof_state.n_observables = config.hwprof.observables.obs_name.n_elements;
   vftrace.hwprof_state.observables = (vftr_observable_t*)malloc(vftrace.hwprof_state.n_observables * sizeof(vftr_observable_t));

   config_hwobservables_t cfg_obs = config.hwprof.observables;
   for (int i = 0; i < vftrace.hwprof_state.n_observables; i++) {
      vftrace.hwprof_state.observables[i].name = cfg_obs.obs_name.values[i];
      vftrace.hwprof_state.observables[i].formula = cfg_obs.formula_expr.values[i];
      if (cfg_obs.unit.set) {
         vftrace.hwprof_state.observables[i].unit = cfg_obs.unit.values[i];
      } else {
         vftrace.hwprof_state.observables[i].unit = NULL;
      }
   }

   char **symbols = (char**)malloc(vftrace.hwprof_state.n_counters * sizeof(char*));
   for (int i = 0; i < vftrace.hwprof_state.n_counters; i++) {
      symbols[i] = config.hwprof.counters.symbol.values[i];
   }

   vftrace.hwprof_state.calculator = vftr_init_calculator (vftrace.hwprof_state.n_observables,
                                        symbols, 
                                        config.hwprof.observables.formula_expr.values);
   free(symbols);

#ifdef _PAPI_AVAIL
   if (vftrace.hwprof_state.hwc_type == HWC_PAPI) vftr_papi_init (&(vftrace.hwprof_state));
#endif
#ifdef _ON_VE
   if (vftrace.hwprof_state.hwc_type == HWC_VE) vftr_veprof_init (vftrace.hwprof_state);
#endif

}

void vftr_hwprof_finalize () {
   if (vftrace.hwprof_state.counters != NULL) free(vftrace.hwprof_state.counters);
   vftrace.hwprof_state.counters = NULL;
   if (vftrace.hwprof_state.observables) {
       free(vftrace.hwprof_state.observables);
       vftrace.hwprof_state.observables = NULL;
   }
   vftr_calculator_free(&vftrace.hwprof_state.calculator);
#ifdef _PAPI_AVAIL
   if (vftrace.hwprof_state.hwc_type == HWC_PAPI) vftr_papi_finalize();
#endif 
}
