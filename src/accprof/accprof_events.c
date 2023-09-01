#include <stdbool.h>

#include "acc_prof.h"

// Helper functions to categorize OpenACC event IDs

bool vftr_accprof_is_data_event (acc_event_t event_type) {
   switch (event_type) {
      case acc_ev_enqueue_upload_start:
      case acc_ev_enqueue_upload_end:
      case acc_ev_enqueue_download_start:
      case acc_ev_enqueue_download_end:
      case acc_ev_enter_data_start:
      case acc_ev_enter_data_end:
      case acc_ev_exit_data_start:
      case acc_ev_exit_data_end:
      case acc_ev_create:
      case acc_ev_delete:
      case acc_ev_alloc:
      case acc_ev_free:
         return true;
      default:
         return false;
   }
}

bool vftr_accprof_is_compute_event (acc_event_t event_type) {
   switch (event_type) {
      case acc_ev_enqueue_launch_start:
      case acc_ev_enqueue_launch_end:
      case acc_ev_compute_construct_start:
      case acc_ev_compute_construct_end:
         return true;
      default:
         return false;
   }
}

bool vftr_accprof_is_h2d_event (acc_event_t event_type) {
   return (event_type == acc_ev_enqueue_upload_start ||
           event_type == acc_ev_enqueue_upload_end);
}

bool vftr_accprof_is_d2h_event (acc_event_t event_type) {
   return (event_type == acc_ev_enqueue_download_start ||
           event_type == acc_ev_enqueue_download_end);
}

bool vftr_accprof_is_ondevice_event (acc_event_t event_type) {
   switch (event_type) {
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
   return event_type == acc_ev_enqueue_launch_start ||
          event_type == acc_ev_enqueue_launch_end;
}

bool vftr_accprof_event_is_defined (acc_event_t event_type) {
  switch (event_type) {
     case acc_ev_enqueue_launch_start:
     case acc_ev_enqueue_launch_end:
     case acc_ev_compute_construct_start:
     case acc_ev_compute_construct_end:
     case acc_ev_enter_data_start:
     case acc_ev_enter_data_end:
     case acc_ev_exit_data_start:
     case acc_ev_exit_data_end:
     case acc_ev_enqueue_upload_start:
     case acc_ev_enqueue_upload_end:
     case acc_ev_enqueue_download_start:
     case acc_ev_enqueue_download_end:
     case acc_ev_update_start:
     case acc_ev_update_end:
     case acc_ev_create:
     case acc_ev_delete:
     case acc_ev_alloc:
     case acc_ev_free:
     case acc_ev_wait_start:
     case acc_ev_wait_end:
        return true;
     default:
        return false;
  }
}

char *vftr_accprof_event_string (acc_event_t event_type) {
  switch (event_type) {
     case acc_ev_enqueue_launch_start:
     case acc_ev_enqueue_launch_end:
        return "launch";
     case acc_ev_compute_construct_start:
     case acc_ev_compute_construct_end:
        return "compute construct";
     case acc_ev_enter_data_start:
     case acc_ev_enter_data_end:
        return "enter_data";
     case acc_ev_exit_data_start:
     case acc_ev_exit_data_end:
        return "exit_data";
     case acc_ev_enqueue_upload_start:
     case acc_ev_enqueue_upload_end:
        return "upload";
     case acc_ev_enqueue_download_start:
     case acc_ev_enqueue_download_end:
        return "download";
     case acc_ev_update_start:
     case acc_ev_update_end:
        return "update";
     case acc_ev_create:
        return "create";
     case acc_ev_delete:
        return "delete";
     case acc_ev_alloc:
        return "alloc";
     case acc_ev_free:
        return "free";
     case acc_ev_wait_start:
     case acc_ev_wait_end:
        return "wait";
     default:
        return "undefined";
  }
}
