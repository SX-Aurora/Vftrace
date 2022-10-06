#include "profiling_types.h"
#include "cupti_event_types.h"
#include "cupti_events.h"
#include "vftrace_state.h"

void vftr_accumulate_cupti_events(profile_t *this_profile) {
   if (vftrace.cupti_state.event_buffer == NULL) return;

   cupti_event_list_t *this_event = vftrace.cupti_state.event_buffer;
   while (this_event != NULL) {
       this_profile->cuptiprof.n_calls += this_event->n_calls;
       this_profile->cuptiprof.t_compute += this_event->t_acc[T_CUDA_COMP];
       this_profile->cuptiprof.t_memcpy += this_event->t_acc[T_CUDA_MEMCP];
       this_profile->cuptiprof.copied_bytes += this_event->memcpy_bytes;
       this_event = this_event->next;
   }
   //printf ("accumulate: %d %f %f\n", cupti_profile.n_calls, cupti_profile.t_compute, cupti_profile.t_memcpy);

   clear_cupti_events();
}
