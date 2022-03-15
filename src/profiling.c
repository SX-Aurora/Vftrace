#include <stdbool.h>

#include "profiling_types.h"

callProfile_t vftr_new_callprofiling() {
   callProfile_t callprof;
   callprof.calls = 0ll;
   callprof.cycles = 0ll;
   callprof.time_usec = 0ll;
   callprof.overhead_time_usec = 0ll;
   return callprof;
}

profile_t vftr_new_profiling() {
   profile_t prof;
   prof.callProf = vftr_new_callprofiling();
   // TODO: include mem profiling
   // TODO: include hardware counter events
   return prof;
}

// accumulate the seperately collected threadprofiling data
// on the stack profile
void vftr_accumulate_callprofiling(bool master, callProfile_t *stackprof,
                                   callProfile_t *threadprof) {
   // TODO: differentiate between master thread and others.
   // data from the other threads is accumulated seperately
   // so that the function time cannot be larger
   // than the execution time of the program
   stackprof->calls += threadprof->calls;
   stackprof->cycles += threadprof->cycles;
   stackprof->time_usec += threadprof->time_usec;
   stackprof->overhead_time_usec += threadprof->overhead_time_usec;
}

void vftr_accumulate_profiling(bool master, profile_t *stackprof,
                               profile_t *threadprof) {
   vftr_accumulate_callprofiling(master,
                                 &(stackprof->callProf),
                                 &(threadprof->callProf));
}

void vftr_callprofiling_free(callProfile_t *callprof_ptr) {
   (void) callprof_ptr;
}

void vftr_profiling_free(profile_t *prof_ptr) {
   (void) prof_ptr;
}
