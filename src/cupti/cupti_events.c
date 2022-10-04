#include "cupti_event_types.h"
#include "vftrace_state.h"

void clear_cupti_events () {
   if (vftrace.cupti_state.event_buffer == NULL) return;
   cupti_event_list_t *this_event = vftrace.cupti_state.event_buffer;
   cupti_event_list_t *next_event;
   while (this_event != NULL) {
      next_event = this_event->next;
      free(this_event);
      this_event = next_event;
   }
   vftrace.cupti_state.event_buffer = NULL;
}
