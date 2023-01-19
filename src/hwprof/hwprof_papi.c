#include <stdio.h>
#include <string.h>

#include <papi.h>

#include "vftrace_state.h"
#include "signal_handling.h"

void vftr_show_papi_components (FILE *fp) {
   int num_components = PAPI_num_components();
   fprintf (fp, "Available components: %d\n", num_components);
   for (int i = 0; i < num_components; i++) {
      const PAPI_component_info_t *cmpinfo = PAPI_get_component_info(i);
      fprintf (fp, "  component: %s\n", cmpinfo->name);
   }
}

void vftr_papi_init (config_t config) {
   int stat;
   if ((stat = PAPI_library_init(PAPI_VER_CURRENT)) != PAPI_VER_CURRENT) {
      fprintf (stderr, "Vftrace Error in PAPI component: Could not initialize (error code %d)\n", stat);
      vftr_abort(0);
   }

   
   if ((stat = PAPI_create_eventset(&vftrace.hwprof_state.papi.eventset)) != PAPI_OK) {
      fprintf (stderr, "Vftrace Error in PAPI component: Could not register eventset (error code %d)\n", stat);
      vftr_abort(0);
   }

   for (int i = 0; i < vftrace.hwprof_state.n_counters; i++) {
      int event_code = PAPI_NATIVE_MASK;
      int stat = PAPI_event_name_to_code ((char*)vftrace.hwprof_state.counters[i].name, &event_code);
      if (stat != PAPI_OK) {
            fprintf (stderr, "Vftrace error in %s: \n", config.config_file_path);
            fprintf (stderr, "No event code exists for %s\n", vftrace.hwprof_state.counters[i].name);
            fprintf (stderr, "Maybe the PAPI component in which it is located is not installed.\n");
            vftr_show_papi_components (stderr);
            vftr_abort(0);
      } else {
         stat = PAPI_add_event(vftrace.hwprof_state.papi.eventset, event_code);
         if (stat == PAPI_ECNFLCT) {
            printf ("No resources for event %s\n", vftrace.hwprof_state.counters[i].name);
            vftr_abort(0);
         } else if (stat != PAPI_OK) {
            printf ("Could not add event %s with error code %d\n",
                     vftrace.hwprof_state.counters[i].name, stat);
            vftr_abort(0);
         }
      }
   }

   if ((stat = PAPI_start (vftrace.hwprof_state.papi.eventset)) != PAPI_OK) {
      fprintf (stderr, "Vftrace Error in PAPI component: Could not start PAPI (error code %d)\n", stat); 
      vftr_abort(0);
   }
}

void vftr_papi_finalize () {
   if (vftrace.hwprof_state.counters != NULL) free(vftrace.hwprof_state.counters);
   vftrace.hwprof_state.counters = NULL;
   PAPI_cleanup_eventset(vftrace.hwprof_state.papi.eventset);
   PAPI_destroy_eventset(&vftrace.hwprof_state.papi.eventset);
}

long long *vftr_get_papi_counters () {
  int n = vftrace.hwprof_state.n_counters;
  long long *counters = (long long *)malloc (n * sizeof(long long));
  int retval = PAPI_read (vftrace.hwprof_state.papi.eventset, counters);
  return counters;
}


