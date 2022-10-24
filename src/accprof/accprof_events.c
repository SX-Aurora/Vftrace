#include <stdbool.h>

#include "acc_prof.h"

bool vftr_accprof_is_data_event (acc_event_t event_type) {
   switch (event_type) {
      case acc_ev_enqueue_upload_start:
      case acc_ev_enqueue_upload_end:
      case acc_ev_enqueue_download_start:
      case acc_ev_enqueue_download_end:
      case acc_ev_create:
      case acc_ev_delete:
      case acc_ev_alloc:
      case acc_ev_free:
         return true;
      default:
         return false;
   }
}

bool vftr_accprof_is_launch_event (acc_event_t event_type) {
   switch (event_type) {
      case acc_ev_enqueue_launch_start:
      case acc_ev_enqueue_launch_end:
         return true;
      default:
         return false;
   }
}
