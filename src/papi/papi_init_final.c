#include <stdio.h>
#include <string.h>

#include <papi.h>

#include "vftrace_state.h"
#include "signal_handling.h"

#include "papi_calculator.h"

void vftr_show_papi_components (FILE *fp) {
   int num_components = PAPI_num_components();
   fprintf (fp, "Available components: %d\n", num_components);
   for (int i = 0; i < num_components; i++) {
      const PAPI_component_info_t *cmpinfo = PAPI_get_component_info(i);
      fprintf (fp, "  component: %s\n", cmpinfo->name);
   }
}

void vftr_papi_init (config_t config) {
   vftrace.papi_state.n_counters = config.papi.counters.hwc_name.n_elements;
   vftrace.papi_state.counters = (vftr_counter_t*)malloc(vftrace.papi_state.n_counters * sizeof(vftr_counter_t));

   for (int i = 0; i < vftrace.papi_state.n_counters; i++) {
      vftrace.papi_state.counters[i].name = config.papi.counters.hwc_name.values[i];
   }

   vftrace.papi_state.n_observables = config.papi.observables.obs_name.n_elements;

   if (PAPI_library_init(PAPI_VER_CURRENT) != PAPI_VER_CURRENT) return;

   
   if (PAPI_create_eventset(&vftrace.papi_state.eventset) != PAPI_OK) return;

   for (int i = 0; i < vftrace.papi_state.n_counters; i++) {
      int event_code = PAPI_NATIVE_MASK;
      int stat = PAPI_event_name_to_code ((char*)vftrace.papi_state.counters[i].name, &event_code);
      if (stat != PAPI_OK) {
            fprintf (stderr, "Vftrace error in %s: \n", config.config_file_path);
            fprintf (stderr, "No event code exists for %s\n", vftrace.papi_state.counters[i].name);
            fprintf (stderr, "Maybe the PAPI component in which it is located is not installed.\n");
            vftr_show_papi_components (stderr);
            vftr_abort(0);
      } else {
         stat = PAPI_add_event(vftrace.papi_state.eventset, event_code);
         if (stat == PAPI_ECNFLCT) {
            printf ("No resources for event %s\n", vftrace.papi_state.counters[i].name);
            vftr_abort(0);
         } else if (stat != PAPI_OK) {
            printf ("Could not add event %s with error code %d\n",
                     vftrace.papi_state.counters[i].name, stat);
            vftr_abort(0);
         }
      }
   }

   char **symbols = (char**)malloc(vftrace.papi_state.n_counters * sizeof(char*));
   for (int i = 0; i < vftrace.papi_state.n_counters; i++) {
      symbols[i] = config.papi.counters.symbol.values[i];
   }

   vftrace.papi_state.calculator = vftr_init_papi_calculator (vftrace.papi_state.n_observables, symbols, 
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
