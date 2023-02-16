#include <stdio.h>
#include <stdlib.h>

#include "vftrace_state.h"
#include "accprof_callbacks.h"

#include "openacc.h"

// Set GPU info. If no GPUs are found, make Vftrace veto callback registration.
// This does not register the OpenACC callbacks. This is done by OpenACC itself.
void vftr_init_accprof () {
   acc_device_t device_type = acc_get_device_type();
   vftrace.accprof_state.n_devices = acc_get_num_devices (device_type);
   if (device_type != acc_device_nvidia || vftrace.accprof_state.n_devices == 0) {
	vftr_veto_accprof_callbacks();
   } else {
        // Currently, only one GPU is detected by OpenACC.
	vftrace.accprof_state.device_names = (const char**)malloc(vftrace.accprof_state.n_devices * sizeof(const char*));
        for (int i = 0; i < vftrace.accprof_state.n_devices; i++) {
           vftrace.accprof_state.device_names[i] = acc_get_property_string (acc_get_device_num(device_type), device_type, acc_property_name);
        }
   }
}

void vftr_finalize_accprof () {
   if (vftrace.accprof_state.veto_callback_registration) return;
   for (int i = 0; i < vftrace.accprof_state.n_devices; i++) {
      free (vftrace.accprof_state.device_names[i]);
   }
   free (vftrace.accprof_state.device_names);
   open_wait_t *this_queue = vftrace.accprof_state.open_wait_queues;
   open_wait_t *next_queue = NULL;
   while (this_queue != NULL) {
      next_queue = this_queue->next;
      free(this_queue);
      this_queue = next_queue;
   }
   vftr_veto_accprof_callbacks(); 
}

void vftr_print_accprof_gpuinfo (FILE *fp) {
   fprintf (fp, "Registered GPUs: %d\n", vftrace.accprof_state.n_devices);
   for (int i = 0; i < vftrace.accprof_state.n_devices; i++) {
      fprintf (fp, "  %d: %s\n", i + 1, vftrace.accprof_state.device_names[i]);
   }
}
