#include <stdlib.h>
#include <stdio.h>

#include "self_profile.h"
#include "callprofiling_types.h"
#include "collated_stack_types.h"
#include "stack_types.h"

#ifdef _MPI
#include <mpi.h>
#endif

callProfile_t vftr_new_callprofiling() {
   SELF_PROFILE_START_FUNCTION;
   callProfile_t prof;
   prof.calls = 0ll;
   prof.time_usec = 0ll;
   prof.time_excl_usec = 0ll;
   prof.overhead_usec = 0ll;
   SELF_PROFILE_END_FUNCTION;
   return prof;
}

void vftr_accumulate_callprofiling(callProfile_t *prof,
                                   int calls,
                                   long long time_usec) {
   prof->calls += calls;
   prof->time_usec += time_usec;
}

void vftr_accumulate_callprofiling_overhead(callProfile_t *prof,
                                            long long overhead_usec) {
   prof->overhead_usec += overhead_usec;
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
   SELF_PROFILE_END_FUNCTION;
}

long long *vftr_get_total_call_overhead(stacktree_t stacktree, int nthreads) {
   SELF_PROFILE_START_FUNCTION;
   // accumulate the hook overhead for each thread separately
   long long *overheads_usec = (long long*) malloc(nthreads*sizeof(long long));
   for (int ithread=0; ithread<nthreads; ithread++) {
      overheads_usec[ithread] = 0ll;
   }

   int nstacks = stacktree.nstacks;
   for (int istack=0; istack<nstacks; istack++) {
      stack_t *stack = stacktree.stacks+istack;
      int nprofs = stack->profiling.nprofiles;
      for (int iprof=0; iprof<nprofs; iprof++) {
         profile_t *prof = stack->profiling.profiles+iprof;
         int threadID = prof->threadID;
         overheads_usec[threadID] += prof->callProf.overhead_usec;
      }
   }
   SELF_PROFILE_END_FUNCTION;
   return overheads_usec;
}

long long vftr_get_total_collated_call_overhead(collated_stacktree_t stacktree) {
   SELF_PROFILE_START_FUNCTION;
   // accumulate the hook overhead for each thread separately
   long long overheads_usec = 0ll;

   int nstacks = stacktree.nstacks;
   for (int istack=0; istack<nstacks; istack++) {
      collated_stack_t *stack = stacktree.stacks+istack;
      collated_profile_t *prof = &(stack->profile);
      overheads_usec += prof->callProf.overhead_usec;
   }
   SELF_PROFILE_END_FUNCTION;
   return overheads_usec;
}

void vftr_callprofiling_free(callProfile_t *callprof_ptr) {
   SELF_PROFILE_START_FUNCTION;
   (void) callprof_ptr;
   SELF_PROFILE_END_FUNCTION;
}

void vftr_print_callprofiling(FILE *fp, callProfile_t callprof) {
   fprintf(fp, "calls: %lld, time(incl/excl): %lld/%lld (overhead: %lld)\n",
           callprof.calls, callprof.time_usec, callprof.time_excl_usec,
           callprof.overhead_usec);
}
