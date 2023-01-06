#include <stdio.h>
#include <string.h>

#include <papi.h>

#include "vftrace_state.h"
#include "signal_handling.h"

#include "papi_calculator.h"

void vftr_papi_init (config_t config) {
   vftrace.papi_state.n_counters = config.papi.counters.hwc_name.n_elements;
   vftrace.papi_state.counters = (vftr_counter_t*)malloc(vftrace.papi_state.n_counters * sizeof(vftr_counter_t));

   for (int i = 0; i < vftrace.papi_state.n_counters; i++) {
      vftrace.papi_state.counters[i].name = config.papi.counters.hwc_name.values[i];
   }

   int n_observables = config.papi.observables.obs_name.n_elements;

   if (PAPI_library_init(PAPI_VER_CURRENT) != PAPI_VER_CURRENT) return;

   int num_components = PAPI_num_components();
   //printf ("Nr. of components: %d\n", num_components);
   //for (int i = 0; i < num_components; i++) {
   //   const PAPI_component_info_t *cmpinfo = PAPI_get_component_info(i);
   //   printf ("component: %s\n", cmpinfo->name);
   //}
   
   if (PAPI_create_eventset(&vftrace.papi_state.eventset) != PAPI_OK) return;

   for (int i = 0; i < vftrace.papi_state.n_counters; i++) {
      int event_code = PAPI_NATIVE_MASK;
      int stat = PAPI_event_name_to_code ((char*)vftrace.papi_state.counters[i].name, &event_code);
      if (stat != PAPI_OK) {
            //int component_type = vftrace.papi_state.counters[i].component_type;
            fprintf (stderr, "Vftrace error in %s: \n", config.config_file_path);
            fprintf (stderr, "  No event code exists for the %s PAPI counter %s.\n",
                     is_native ? "native" : "preset", 
                     vftrace.papi_state.counters[i].name);
            fprintf (stderr, "  Check \"%s\" if this counter exists on the platform"
                             "you are running the application on.\n",
                     is_native ? "papi_native_avail" : "papi_avail");
            vftr_abort(0);
      } else {
         PAPI_add_event(vftrace.papi_state.eventset, event_code);
      }
   }

   //int n_variables = config.papi.counters.native_name.n_elements
   //                + config.papi.counters.preset_name.n_elements
   //                + config.papi.counters.appio_name.n_elements;
   char **symbols = (char**)malloc(vftrace.papi_state.n_counters * sizeof(char*));
   for (int i = 0; i < vftrace.papi_state.n_counters; i++) {
      symbols[i] = config.papi.counters.symbol.values[i];
   }

   vftrace.papi_state.calculator = vftr_init_papi_calculator (n_observables, symbols, 
                                        config.papi.observables.formula_expr.values);
   free(symbols);

   PAPI_start (vftrace.papi_state.eventset);
}

void vftr_papi_finalize () {
   if (vftrace.papi_state.counters != NULL) free(vftrace.papi_state.counters);
   vftrace.papi_state.counters = NULL;
   PAPI_cleanup_eventset(vftrace.papi_state.eventset);
   PAPI_destroy_eventset(&vftrace.papi_state.eventset);
}
