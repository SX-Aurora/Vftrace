#include <stdlib.h>
#include <stdio.h>

#include "self_profile.h"
#include "callprofiling_types.h"
#include "stack_types.h"

#ifdef _MPI
#include <mpi.h>
#endif

callprofile_t vftr_new_callprofiling() {
   SELF_PROFILE_START_FUNCTION;
   callprofile_t prof;
   prof.calls = 0ll;
   prof.time_nsec = 0ll;
   prof.time_excl_nsec = 0ll;
   prof.overhead_nsec = 0ll;
   SELF_PROFILE_END_FUNCTION;
   return prof;
}

void vftr_accumulate_callprofiling(callprofile_t *prof,
                                   int calls,
                                   long long time_nsec) {
   prof->calls += calls;
   prof->time_nsec += time_nsec;
}

void vftr_accumulate_callprofiling_overhead(callprofile_t *prof,
                                            long long overhead_nsec) {
   prof->overhead_nsec += overhead_nsec;
}

callprofile_t vftr_add_callprofiles(callprofile_t profA, callprofile_t profB) {
   callprofile_t profC;
   profC.calls = profA.calls + profB.calls;
   profC.time_nsec = profA.time_nsec + profB.time_nsec;
   profC.time_excl_nsec = profA.time_excl_nsec + profB.time_excl_nsec;
   profC.overhead_nsec = profA.overhead_nsec + profB.overhead_nsec;
   return profC;
}

void vftr_update_stacks_exclusive_time(stacktree_t *stacktree_ptr) {
   SELF_PROFILE_START_FUNCTION;
   int nstacks = stacktree_ptr->nstacks;
   stack_t *stacks = stacktree_ptr->stacks;
   // exclusive time for init is 0, therefore it does not need to be computed.
   for (int istack=1; istack<nstacks; istack++) {
      stack_t *mystack = stacks + istack;
      // need to go over the calling profiles threadwise
      for (int iprof=0; iprof<mystack->profiling.nprofiles; iprof++) {
         profile_t *myprof = mystack->profiling.profiles+iprof;
         myprof->callprof.time_excl_nsec = myprof->callprof.time_nsec;
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
               myprof->callprof.time_excl_nsec -= calleeprof->callprof.time_nsec;
            }
         }
      }
   }
   SELF_PROFILE_END_FUNCTION;
}

long long *vftr_get_total_call_overhead(stacktree_t stacktree, int nthreads) {
   SELF_PROFILE_START_FUNCTION;
   // accumulate the hook overhead for each thread separately
   long long *overheads_nsec = (long long*) malloc(nthreads*sizeof(long long));
   for (int ithread=0; ithread<nthreads; ithread++) {
      overheads_nsec[ithread] = 0ll;
   }

   int nstacks = stacktree.nstacks;
   for (int istack=0; istack<nstacks; istack++) {
      stack_t *stack = stacktree.stacks+istack;
      int nprofs = stack->profiling.nprofiles;
      for (int iprof=0; iprof<nprofs; iprof++) {
         profile_t *prof = stack->profiling.profiles+iprof;
         int threadID = prof->threadID;
         overheads_nsec[threadID] += prof->callprof.overhead_nsec;
      }
   }
   SELF_PROFILE_END_FUNCTION;
   return overheads_nsec;
}

void vftr_callprofiling_free(callprofile_t *callprof_ptr) {
   SELF_PROFILE_START_FUNCTION;
   (void) callprof_ptr;
   SELF_PROFILE_END_FUNCTION;
}

void vftr_print_callprofiling(FILE *fp, callprofile_t callprof) {
   fprintf(fp, "calls: %lld, time(incl/excl): %lld/%lld (overhead: %lld)\n",
           callprof.calls, callprof.time_nsec, callprof.time_excl_nsec,
           callprof.overhead_nsec);
}
