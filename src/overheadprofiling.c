#include <stdbool.h>

#include "vftrace_state.h"
#include "threads.h"
#include "threadstacks.h"
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

#ifdef _MPI
void vftr_accumulate_mpi_overheadprofiling(overheadProfile_t *prof,
                                           long long overhead_usec) {
   prof->mpi_usec += overhead_usec;
}
#endif

#ifdef _OMP
void vftr_accumulate_omp_overheadprofiling(overheadProfile_t *prof,
                                           long long overhead_usec) {
   prof->omp_usec += overhead_usec;
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
