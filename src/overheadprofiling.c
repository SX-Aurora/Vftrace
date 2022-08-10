#include <stdlib.h>
#include <stdbool.h>

#include "vftrace_state.h"
#include "threads.h"
#include "threadstacks.h"
#include "stack_types.h"
#include "stacks.h"
#include "profiling.h"
#include "overheadprofiling_types.h"

overheadProfile_t vftr_new_overheadprofiling() {
   overheadProfile_t prof;
   prof.hook_usec = 0ll;
#ifdef _MPI
   prof.mpi_usec = 0ll;
#endif
#ifdef _OMP
   prof.omp_usec = 0ll;
#endif
   return prof;
}

void vftr_accumulate_hook_overheadprofiling(overheadProfile_t *prof,
                                            long long overhead_usec) {
   prof->hook_usec += overhead_usec;
}

long long *vftr_get_total_hook_overhead(stacktree_t stacktree, int nthreads) {
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
         overheads_usec[threadID] += prof->overheadProf.hook_usec;
      }
   }
   return overheads_usec;
}

#ifdef _MPI
void vftr_accumulate_mpi_overheadprofiling(overheadProfile_t *prof,
                                           long long overhead_usec) {
   prof->mpi_usec += overhead_usec;
}

long long *vftr_get_total_mpi_overhead(stacktree_t stacktree, int nthreads) {
   // accumulate the mpi overhead for each thread separately
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
         overheads_usec[threadID] += prof->overheadProf.mpi_usec;
      }
   }
   return overheads_usec;
}
#endif

#ifdef _OMP
void vftr_accumulate_omp_overheadprofiling(overheadProfile_t *prof,
                                           long long overhead_usec) {
   prof->omp_usec += overhead_usec;
}

long long *vftr_get_total_omp_overhead(stacktree_t stacktree, int nthreads) {
   // accumulate the omp overhead for each thread separately
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
         overheads_usec[threadID] += prof->overheadProf.omp_usec;
      }
   }
   return overheads_usec;
}
#endif

void vftr_overheadprofiling_free(overheadProfile_t *overheadprof_ptr) {
   (void) overheadprof_ptr;
}

overheadProfile_t *vftr_get_my_overheadProfile(vftrace_t vftrace) {
   thread_t *my_thread = vftr_get_my_thread(&(vftrace.process.threadtree));
   threadstack_t *my_threadstack = vftr_get_my_threadstack(my_thread);
   stack_t *my_stack = vftrace.process.stacktree.stacks+my_threadstack->stackID;
   profile_t *my_profile = vftr_get_my_profile(my_stack, my_thread);
   return &(my_profile->overheadProf);
}

void vftr_print_overheadprofiling(FILE *fp, overheadProfile_t overheadprof) {
   fprintf(fp, "hooks: %lld, mpi: %lld, omp: %lld\n",
           overheadprof.hook_usec,
#ifdef _MPI
           overheadprof.mpi_usec,
#else
           0ll,
#endif
#ifdef _OMP
           overheadprof.omp_usec,
#else
           0ll
#endif
         );
}
