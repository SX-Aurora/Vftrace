#include <stdlib.h>
#include <stdbool.h>

#include <string.h>

#include "self_profile.h"
#include "vftrace_state.h"
#include "process_types.h"
#include "environment_types.h"
#include "ompprofiling_types.h"

#include "range_expand.h"
#include "search.h"

ompprofile_t vftr_new_ompprofiling() {
   ompprofile_t prof;
   prof.overhead_nsec = 0ll;

   return prof;
}

void vftr_accumulate_ompprofiling_overhead(ompprofile_t *prof,
                                           long long overhead_nsec) {
   prof->overhead_nsec += overhead_nsec;
}

void vftr_ompprofiling_free(ompprofile_t *prof_ptr) {
   (void) prof_ptr;
}

long long *vftr_get_total_omp_overhead(stacktree_t stacktree, int nthreads) {
   SELF_PROFILE_START_FUNCTION;
   // accumulate the omp overhead for each thread separately
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
         overheads_nsec[threadID] += prof->ompprof.overhead_nsec;
      }
   }
   SELF_PROFILE_END_FUNCTION;
   return overheads_nsec;
}

long long vftr_get_total_collated_omp_overhead(collated_stacktree_t stacktree) {
   SELF_PROFILE_START_FUNCTION;
   long long overheads_nsec = 0ll;

   int nstacks = stacktree.nstacks;
   for (int istack=0; istack<nstacks; istack++) {
      collated_stack_t *stack = stacktree.stacks+istack;
      collated_profile_t *prof = &(stack->profile);
      overheads_nsec += prof->ompprof.overhead_nsec;
   }
   SELF_PROFILE_END_FUNCTION;
   return overheads_nsec;
}

void vftr_print_ompprofiling(FILE *fp, ompprofile_t ompprof) {
//   fprintf(fp, "overhead: %ldnmsg: %lld/%lld, msgsize: %lld/%lld\n",
//           ompprof.nsendmessages,
//           ompprof.nrecvmessages,
//           ompprof.send_bytes,
//           mpiprof.recv_bytes);
}
