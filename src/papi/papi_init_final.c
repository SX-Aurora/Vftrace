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

   vftrace.papi_state.calculator = vftr_init_papi_calculator (config, n_variables, n_observables,
                                   symbols,
                                   config.papi.observables.formula_expr.values);
   free(symbols);

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
