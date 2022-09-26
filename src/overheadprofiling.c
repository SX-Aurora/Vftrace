#include <stdlib.h>
#include <stdbool.h>

#include "self_profile.h"
#include "vftrace_state.h"
#include "threads.h"
#include "threadstacks.h"
#include "stack_types.h"
#include "stacks.h"
#include "profiling.h"
#include "overheadprofiling_types.h"

overheadprofile_t vftr_new_overheadprofiling() {
   SELF_PROFILE_START_FUNCTION;
   overheadprofile_t prof;
   prof.hook_nsec = 0ll;
#ifdef _MPI
   prof.mpi_nsec = 0ll;
#endif
#ifdef _OMP
   prof.omp_nsec = 0ll;
#endif
   SELF_PROFILE_END_FUNCTION;
   return prof;
}

void vftr_accumulate_hook_overheadprofiling(overheadprofile_t *prof,
                                            long long overhead_nsec) {
   prof->hook_nsec += overhead_nsec;
}

long long *vftr_get_total_hook_overhead(stacktree_t stacktree, int nthreads) {
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
         overheads_nsec[threadID] += prof->overheadprof.hook_nsec;
      }
   }
   SELF_PROFILE_END_FUNCTION;
   return overheads_nsec;
}

#ifdef _MPI
void vftr_accumulate_mpi_overheadprofiling(overheadprofile_t *prof,
                                           long long overhead_nsec) {
   prof->mpi_nsec += overhead_nsec;
}

long long *vftr_get_total_mpi_overhead(stacktree_t stacktree, int nthreads) {
   SELF_PROFILE_START_FUNCTION;
   // accumulate the mpi overhead for each thread separately
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
         overheads_nsec[threadID] += prof->overheadprof.mpi_nsec;
      }
   }
   SELF_PROFILE_END_FUNCTION;
   return overheads_nsec;
}
#endif

#ifdef _OMP
void vftr_accumulate_omp_overheadprofiling(overheadprofile_t *prof,
                                           long long overhead_nsec) {
   prof->omp_nsec += overhead_nsec;
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
         overheads_nsec[threadID] += prof->overheadprof.omp_nsec;
      }
   }
   SELF_PROFILE_END_FUNCTION;
   return overheads_nsec;
}
#endif

void vftr_overheadprofiling_free(overheadprofile_t *overheadprof_ptr) {
   (void) overheadprof_ptr;
}

overheadprofile_t *vftr_get_my_overheadprofile(vftrace_t vftrace) {
   SELF_PROFILE_START_FUNCTION;
   thread_t *my_thread = vftr_get_my_thread(&(vftrace.process.threadtree));
   threadstack_t *my_threadstack = vftr_get_my_threadstack(my_thread);
   stack_t *my_stack = vftrace.process.stacktree.stacks+my_threadstack->stackID;
   profile_t *my_profile = vftr_get_my_profile(my_stack, my_thread);
   SELF_PROFILE_END_FUNCTION;
   return &(my_profile->overheadprof);
}

void vftr_print_overheadprofiling(FILE *fp, overheadprofile_t overheadprof) {
   SELF_PROFILE_START_FUNCTION;
   fprintf(fp, "hooks: %lld, mpi: %lld, omp: %lld\n",
           overheadprof.hook_nsec,
#ifdef _MPI
           overheadprof.mpi_nsec,
#else
           0ll,
#endif
#ifdef _OMP
           overheadprof.omp_nsec,
#else
           0ll
#endif
         );
   SELF_PROFILE_END_FUNCTION;
}
