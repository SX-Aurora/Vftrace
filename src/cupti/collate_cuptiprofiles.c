#include "collated_stack_types.h"
#include "cuptiprofiling_types.h"

#include <stdio.h>

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
                collcuptiprof->t_vftr = copy_cuptiprof.t_vftr;
                collcuptiprof->memcpy_bytes[0] = copy_cuptiprof.memcpy_bytes[0];
                collcuptiprof->memcpy_bytes[1] = copy_cuptiprof.memcpy_bytes[1];
                // Start and stop events are irrelevant for the collated profile.
        }
}


