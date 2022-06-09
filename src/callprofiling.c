#include <stdio.h>

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

void vftr_update_stacks_exclusive_time(stacktree_t *stacktree_ptr) {
   int nstacks = stacktree_ptr->nstacks;
   stack_t *stacks = stacktree_ptr->stacks;
   // exclusive time for init is 0, therefore it does not need to be computed.
   for (int istack=1; istack<nstacks; istack++) {
      stack_t *mystack = stacks + istack;
      // need to go over the calling profiles threadwise
      for (int iprof=0; iprof<mystack->profiling.nprofiles; iprof++) {
         profile_t *myprof = mystack->profiling.profiles+iprof;
         myprof->callProf.time_excl_usec = myprof->callProf.time_usec;
         // subtract the time spent in the callees
         for (int icallee=0; icallee<mystack->ncallees; icallee++) {
            int calleeID = mystack->callees[icallee];
            stack_t *calleestack = stacks+calleeID;
            // search for a thread matching profile in the callee profiles
            int calleprofID = -1;
            for (int jprof=0; jprof<calleestack->profiling.nprofiles; jprof++) {
               profile_t *calleeprof = calleestack->profiling.profiles+jprof;
               if (myprof->threadID == calleeprof->threadID) {
                  calleprofID = jprof;
                  break;
               }
            }
            // a matching callee profile was found
            if (calleprofID >= 0) {
               profile_t *calleeprof = calleestack->profiling.profiles+calleprofID;
               myprof->callProf.time_excl_usec -= calleeprof->callProf.time_usec;
            }
         }
      }
   }
}

void vftr_callprofiling_free(callProfile_t *callprof_ptr) {
   (void) callprof_ptr;
}

void vftr_print_callprofiling(FILE *fp, callProfile_t callprof) {
   fprintf(fp, "calls: %lld, cycles: %lld, time(incl/excl): %lld/%lld\n",
           callprof.calls, callprof.cycles,
           callprof.time_usec, callprof.time_excl_usec);
}
