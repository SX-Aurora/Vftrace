#include <stdlib.h>
#include <stdio.h>

#include "self_profile.h"
#include "collated_callprofiling_types.h"
#include "collated_stack_types.h"
#include "stack_types.h"

#ifdef _MPI
#include <mpi.h>
#endif

collated_callProfile_t vftr_new_collated_callprofiling() {
   SELF_PROFILE_START_FUNCTION;
   collated_callProfile_t prof;
   prof.calls = 0ll;
   prof.time_usec = 0ll;
   prof.time_excl_usec = 0ll;
   prof.overhead_usec = 0ll;
   prof.on_nranks = 0;
   prof.max_on_rank = 0;
   prof.min_on_rank = 0;
   prof.average_time_usec = 0ll;
   prof.max_time_usec = 0ll;
   prof.min_time_usec = 0ll;
   prof.max_imbalance = 0.0;
   prof.max_imbalance_on_rank = 0;
   SELF_PROFILE_END_FUNCTION;
   return prof;
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

void vftr_collated_callprofiling_free(collated_callProfile_t *callprof_ptr) {
   SELF_PROFILE_START_FUNCTION;
   (void) callprof_ptr;
   SELF_PROFILE_END_FUNCTION;
}

void vftr_print_collated_callprofiling(FILE *fp, collated_callProfile_t callprof) {
   fprintf(fp, "calls: %lld, time(incl/excl): %lld/%lld (overhead: %lld)\n",
           callprof.calls, callprof.time_usec, callprof.time_excl_usec,
           callprof.overhead_usec);
}
