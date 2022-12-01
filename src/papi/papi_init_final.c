#include <stdio.h>
#include <string.h>

#include <papi.h>

#include "vftrace_state.h"

#include "papi_calculator.h"


void vftr_papi_init (config_t config) {
   printf ("Counters: \n");
   for (int i = 0; i < config.papi.counters.native_name.n_elements; i++) {
      printf ("  native: %s\n", config.papi.counters.native_name.values[i]);
      printf ("  symbol: %s\n", config.papi.counters.symbol.values[i]);
   }
   printf ("Observables: \n");
   for (int i = 0; i < config.papi.observables.obs_name.n_elements; i++) {
      printf ("  name: %s\n", config.papi.observables.obs_name.values[i]);
      printf ("  formula: %s\n", config.papi.observables.formula_expr.values[i]);
      printf ("  unit: %s\n", config.papi.observables.unit.values[i]);
   }

   int n_observables = config.papi.observables.obs_name.n_elements;

   if (PAPI_library_init(PAPI_VER_CURRENT) != PAPI_VER_CURRENT) return;
   
   if (PAPI_create_eventset(&vftrace.papi_state.eventset) != PAPI_OK) return;

   for (int i = 0; i < config.papi.counters.native_name.n_elements; i++) {
      char *counter_name = config.papi.counters.native_name.values[i];
      int event_code;
      int stat = PAPI_event_name_to_code (counter_name, &event_code);
      if (stat != PAPI_OK) {
            fprintf (stderr, "Vftrace error in %s: \n", config.config_file_path);
            fprintf (stderr, "  No event code exists for the native PAPI counter %s.\n", counter_name);
            fprintf (stderr, "  Check \"papi_native_avail\" if this counter exists on the platform"
                             "you are running the application on.\n");
            abort();
      } else {
         PAPI_add_event(vftrace.papi_state.eventset, event_code);
      }
   }

   int n_variables = config.papi.counters.native_name.n_elements;
   char **symbols = (char**)malloc(n_variables * sizeof(char*));
   for (int i = 0; i < n_variables; i++) {
      symbols[i] = config.papi.counters.symbol.values[i];
   }

   printf ("INIT calc: %d\n", n_variables);
   vftrace.papi_state.calculator = vftr_init_papi_calculator (config, n_variables, n_observables,
                                   symbols,
                                   config.papi.observables.formula_expr.values);
   free(symbols);

   vftr_print_papi_calculator_state (vftrace.papi_state.calculator);

   //int event_code = 0 | PAPI_PRESET_MASK;
   //int n_events = 0;
   //PAPI_event_info_t info;
   //do {
   //   PAPI_get_event_info (event_code, &info);
   //   if (info.count > 0) n_events++; 
   //} while (PAPI_enum_event (&event_code, false) == PAPI_OK);
   //

   //int *event_codes = (int*)malloc (n_events * sizeof(int));
   //bool *event_activated = (bool*)malloc (n_events * sizeof(bool));
   //char **event_names = (char**)malloc (n_events * sizeof(char*));
   //char **event_units = (char**)malloc (n_events * sizeof(char*));
   //char **event_descriptions = (char**)malloc (n_events * sizeof(char*));

   //vftrace.papi_state.n_available_events = n_events;
   //n_events = 0;

   //event_code = 0 | PAPI_PRESET_MASK;

   //do {
   //   PAPI_get_event_info (event_code, &info);
   //   if (info.count == 0) continue;
   //   event_codes[n_events] = event_code;
   //   //event_activated[n_events] = info.count > 0;
   //   event_names[n_events] = strdup(info.symbol);
   //   event_units[n_events] = strdup(info.units);
   //   event_descriptions[n_events] = strdup(info.long_descr);
   //   n_events++; 
   //} while (PAPI_enum_event (&event_code, false) == PAPI_OK);

   //for (int i = 0; i < vftrace.papi_state.n_available_events; i++) {
   //   PAPI_add_event(vftrace.papi_state.eventset, event_codes[i]);
   //} 

   //vftrace.papi_state.event_codes = event_codes;
   ////vftrace.papi_state.event_activated = event_activated;
   //vftrace.papi_state.event_names = event_names;
   //vftrace.papi_state.event_units = event_units;
   //vftrace.papi_state.event_descriptions = event_descriptions;

   PAPI_start (vftrace.papi_state.eventset);
}

void vftr_papi_show_avail_events (FILE *fp) {
   //fprintf (fp, "Number of available HW events: %d\n", vftrace.papi_state.n_available_events);
   //for (int i = 0; i < vftrace.papi_state.n_available_events; i++) {
   //   fprintf (fp, "%u: %s\n", vftrace.papi_state.event_codes[i], vftrace.papi_state.event_names[i]);
   //}





  
}
