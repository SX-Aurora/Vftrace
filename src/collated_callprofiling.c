#include <stdlib.h>
#include <stdio.h>

#include "self_profile.h"
#include "collated_callprofiling_types.h"
#include "collated_stack_types.h"
#include "stack_types.h"

#ifdef _MPI
#include <mpi.h>
#endif

collated_callprofile_t vftr_new_collated_callprofiling() {
   SELF_PROFILE_START_FUNCTION;
   collated_callprofile_t prof;
   prof.calls = 0ll;
   prof.time_nsec = 0ll;
   prof.time_excl_nsec = 0ll;
   prof.overhead_nsec = 0ll;
   prof.on_nranks = 0;
   prof.max_on_rank = 0;
   prof.min_on_rank = 0;
   prof.average_time_nsec = 0ll;
   prof.max_time_nsec = 0ll;
   prof.min_time_nsec = 0ll;
   prof.max_imbalance = 0.0;
   prof.max_imbalance_on_rank = 0;
   SELF_PROFILE_END_FUNCTION;
   return prof;
}

collated_callprofile_t vftr_add_collated_callprofiles(collated_callprofile_t profA,
                                                      collated_callprofile_t profB) {
   collated_callprofile_t profC;
   profC.calls = profA.calls + profB.calls;
   profC.time_nsec = profA.time_nsec + profB.time_nsec;
   profC.time_excl_nsec = profA.time_excl_nsec + profB.time_excl_nsec;
   profC.overhead_nsec = profA.overhead_nsec + profB.overhead_nsec;
   // This is currently exclusively used for name grouped stacks
   // Therefore, it does not make sense to care about the imbalances
   return profC;
}

long long vftr_get_total_collated_call_overhead(collated_stacktree_t stacktree) {
   SELF_PROFILE_START_FUNCTION;
   // accumulate the hook overhead for each thread separately
   long long overheads_nsec = 0ll;

   int nstacks = stacktree.nstacks;
   for (int istack=0; istack<nstacks; istack++) {
      collated_stack_t *stack = stacktree.stacks+istack;
      collated_profile_t *prof = &(stack->profile);
      overheads_nsec += prof->callprof.overhead_nsec;
   }
   SELF_PROFILE_END_FUNCTION;
   return overheads_nsec;
}

void vftr_collated_callprofiling_free(collated_callprofile_t *callprof_ptr) {
   SELF_PROFILE_START_FUNCTION;
   (void) callprof_ptr;
   SELF_PROFILE_END_FUNCTION;
}

void vftr_print_collated_callprofiling(FILE *fp, collated_callprofile_t callprof) {
   fprintf(fp, "calls: %lld, time(incl/excl): %lld/%lld (overhead: %lld)\n",
           callprof.calls, callprof.time_nsec, callprof.time_excl_nsec,
           callprof.overhead_nsec);
}

void vftr_print_calltime_imbalances(FILE *fp, collated_callprofile_t callprof) {
   fprintf(fp, "avg: %lldus, min/max=%lldus(on %d)/%lldus(on %d), imb=%6.2lf%% on %d\n",
           callprof.average_time_nsec,
           callprof.max_time_nsec, callprof.max_on_rank,
           callprof.min_time_nsec, callprof.min_on_rank,
           callprof.max_imbalance, callprof.max_imbalance_on_rank);
}
