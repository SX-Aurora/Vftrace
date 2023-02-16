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
      vftr_stack_t *stack = stacktree_ptr->stacks + istack;
      int i_collstack = stack->gid;
      collated_stack_t *collstack = collstacktree_ptr->stacks + i_collstack;
      // OpenACC is only supported for one thread (i_prof = 0).
      accprofile_t copy_accprof = stack->profiling.profiles[0].accprof;
      accprofile_t *collaccprof = &(collstack->profile.accprof);
      
      collaccprof->region_id = copy_accprof.region_id;
      collaccprof->event_type = copy_accprof.event_type;
      collaccprof->line_start = copy_accprof.line_start;
      collaccprof->line_end = copy_accprof.line_end;
      collaccprof->copied_bytes = copy_accprof.copied_bytes;
      collaccprof->source_file = copy_accprof.source_file;
      collaccprof->var_name = copy_accprof.var_name;
      collaccprof->func_name = copy_accprof.func_name;
      collaccprof->kernel_name = copy_accprof.kernel_name;
      collaccprof->overhead_nsec = copy_accprof.overhead_nsec;
      collaccprof->region_id = copy_accprof.region_id;
   }
}

