#include <stdio.h>
#include <string.h>

#include <papi.h>

#include "vftrace_state.h"

#include "papi_calculator.h"

void vftr_papi_init (config_t config) {
   int n_native = config.papi.counters.native_name.n_elements;
   int n_preset = config.papi.counters.preset_name.n_elements;
   vftrace.papi_state.n_counters = n_native + n_preset;

   vftrace.papi_state.counters = (vftr_counter_t*)malloc(vftrace.papi_state.n_counters * sizeof(vftr_counter_t));
   for (int i = 0; i < n_native; i++) {
      vftrace.papi_state.counters[i].name = config.papi.counters.native_name.values[i];
      vftrace.papi_state.counters[i].is_native = true;
   } 

   for (int i = 0; i < n_preset; i++) {
      vftrace.papi_state.counters[n_native + i].name = config.papi.counters.preset_name.values[i];
      vftrace.papi_state.counters[n_native + i].is_native = false;
   }

   int n_observables = config.papi.observables.obs_name.n_elements;

   if (PAPI_library_init(PAPI_VER_CURRENT) != PAPI_VER_CURRENT) return;
   
   if (PAPI_create_eventset(&vftrace.papi_state.eventset) != PAPI_OK) return;

   for (int i = 0; i < n_native + n_preset; i++) {
      int event_code;
      int stat = PAPI_event_name_to_code (vftrace.papi_state.counters[i].name, &event_code);
      if (stat != PAPI_OK) {
            bool is_native = vftrace.papi_state.counters[i].is_native;
            fprintf (stderr, "Vftrace error in %s: \n", config.config_file_path);
            fprintf (stderr, "  No event code exists for the %s PAPI counter %s.\n",
                     is_native ? "native" : "preset", 
                     vftrace.papi_state.counters[i].name);
            fprintf (stderr, "  Check \"%s\" if this counter exists on the platform"
                             "you are running the application on.\n",
                     is_native ? "papi_native_avail" : "papi_avail");
            abort();
      } else {
         PAPI_add_event(vftrace.papi_state.eventset, event_code);
      }
   }

   int n_variables = config.papi.counters.native_name.n_elements + config.papi.counters.preset_name.n_elements;
   char **symbols = (char**)malloc(n_variables * sizeof(char*));
   for (int i = 0; i < n_variables; i++) {
      symbols[i] = config.papi.counters.symbol.values[i];
   }

   vftrace.papi_state.calculator = vftr_init_papi_calculator (n_variables, n_observables,
                                   symbols,
                                   config.papi.observables.formula_expr.values);
   free(symbols);

   PAPI_start (vftrace.papi_state.eventset);
}
