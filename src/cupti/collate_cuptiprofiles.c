#include "collated_stack_types.h"
#include "cuptiprofiling_types.h"

// Currently, the CUPTI interface is only supported for
// one MPI process and one OMP thread. Therefore, collating
// the profiles just comes down to copying the profile from
// the one stack which exists.
void vftr_collate_cuptiprofiles (collated_stacktree_t *collstacktree_ptr, 
  				 stacktree_t *stacktree_ptr,
 				 int myrank, int nranks, int *nremote_profiles) {
	for (int istack = 0; istack < stacktree_ptr->nstacks; istack++) {
		stack_t *stack = stacktree_ptr->stacks + istack;
		int i_collstack = stack->gid;
		collated_stack_t *collstack = collstacktree_ptr->stacks + i_collstack;
                // CUPTI is only supported for one thread (i_prof = 0).
                cuptiprofile_t copy_cuptiprof = stack->profiling.profiles[0].cuptiprof;
		cuptiprofile_t *collcuptiprof = &(collstack->profile.cuptiprof);
		
		collcuptiprof->cbid = copy_cuptiprof.cbid;
                collcuptiprof->n_calls = copy_cuptiprof.n_calls;
                collcuptiprof->t_ms = copy_cuptiprof.t_ms;
                collcuptiprof->memcpy_bytes[0] = copy_cuptiprof.memcpy_bytes[0];
                collcuptiprof->memcpy_bytes[1] = copy_cuptiprof.memcpy_bytes[1];
 		collcuptiprof->overhead_nsec = copy_cuptiprof.overhead_nsec;
                // Start and stop events are irrelevant for the collated profile.
        }
}


