#include <stdio.h>
#include <stdbool.h>

#include "callprofiling_types.h"
#include "stack_types.h"

callProfile_t vftr_new_callprofiling() {
   callProfile_t prof;
   prof.calls = 0ll;
   prof.cycles = 0ll;
   prof.time_usec = 0ll;
   prof.time_excl_usec = 0ll;
   return prof;
}

void vftr_accumulate_callprofiling(callProfile_t *prof,
                                   int calls,
                                   long long cycles,
                                   long long time_usec) {
   prof->calls += calls;
   prof->cycles += cycles;
   prof->time_usec += time_usec;
}

void vftr_update_stacks_exclusive_time(int nstacks, stack_t *stacks) {
//TODO    // exclusive time for init is 0, therefore it does not need to be computed.
//TODO    for (int istack=1; istack<nstacks; istack++) {
//TODO       stack_t *mystack = stacks + istack;
//TODO       long long exclusive_time = mystack->profiling.callProf.time_usec;
//TODO       // subtract the time spent in the callees
//TODO       for (int icallee=0; icallee<mystack->ncallees; icallee++) {
//TODO          int calleeID = mystack->callees[icallee];
//TODO          exclusive_time -= stacks[calleeID].profiling.callProf.time_usec;
//TODO       }
//TODO       mystack->profiling.callProf.time_excl_usec = exclusive_time;
//TODO    }
}

void vftr_callprofiling_free(callProfile_t *callprof_ptr) {
   (void) callprof_ptr;
}

void vftr_print_callprofiling(FILE *fp, callProfile_t callprof) {
   fprintf(fp, "calls: %lld, cycles: %lld, time(incl/excl): %lld/%lld\n",
           callprof.calls, callprof.cycles,
           callprof.time_usec, callprof.time_excl_usec);
}
