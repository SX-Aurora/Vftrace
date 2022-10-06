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
		cuptiprofile_t *collcuptiprof = &(collstack->profile.cuptiprof);
		
                // CUPTI is only supported for one thread (i_prof = 0).
	        cuptiprofile_t *cuptiprof_to_copy = &(stack->profiling.profiles[0].cuptiprof);
                collcuptiprof->events = cuptiprof_to_copy->events;
                // Start and stop events are irrelevant for the collated profile.
        }
}


