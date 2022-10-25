#include "collated_stack_types.h"
#include "accprofiling_types.h"

// Currently, OpenACC profiling is only supported for one MPI process
// and one OMP thread. Therefore, collating the profiles just comes
// down to copying the profile from the one stack which exists.
void vftr_collate_accprofiles (collated_stacktree_t *collstacktree_ptr,
                               stacktree_t *stacktree_ptr,
                               int myrank, int nranks, int *nremote_profiles) {
   (void)myrank;
   (void)nranks;
   (void)nremote_profiles;

   for (int istack = 0; istack < stacktree_ptr->nstacks; istack++) {
      stack_t *stack = stacktree_ptr->stacks + istack;
      int i_collstack = stack->gid;
      collated_stack_t *collstack = collstacktree_ptr->stacks + i_collstack;
      // OpenACC is only supported for one thread (i_prof = 0).
      accprofile_t copy_accprof = stack->profiling.profiles[0].accprof;
      accprofile_t *collaccprof = &(collstack->profile.accprof);
      
      collaccprof->event_type = copy_accprof.event_type;
      collaccprof->copied_bytes = copy_accprof.copied_bytes;
      collaccprof->var_name = copy_accprof.var_name;
      collaccprof->kernel_name = copy_accprof.kernel_name;
   }
}

