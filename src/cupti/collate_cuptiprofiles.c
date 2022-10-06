#include "collated_stack_types.h"
#include "cuptiprofiling_types.h"

void vftr_collate_cuptiprofiles (collated_stacktree_t *collstacktree_ptr, 
  				 stacktree_t *stacktree_ptr,
 				 int myrank, int nranks, int *nremote_profiles) {
	for (int istack = 0; istack < stacktree_ptr->nstacks; istack++) {
		stack_t *stack = stacktree_ptr->stacks + istack;
		int i_collstack = stack->gid;
		
		collated_stack_t *collstack = collstacktree_ptr->stacks + i_collstack;
		cuptiprofile_t *collcuptiprof = &(collstack->profile.cuptiprof);
		
		collcuptiprof->n_calls = 0;
		collcuptiprof->t_compute = 0;
		collcuptiprof->t_memcpy = 0;
		collcuptiprof->copied_bytes = 0;

		for (int i_prof = 0; i_prof < stack->profiling.nprofiles; i_prof++) {
			cuptiprofile_t *cuptiprof = &(stack->profiling.profiles[i_prof].cuptiprof);
			collcuptiprof->n_calls += cuptiprof->n_calls;
			collcuptiprof->t_compute += cuptiprof->t_compute;
			collcuptiprof->t_memcpy += cuptiprof->t_memcpy;
			collcuptiprof->copied_bytes += cuptiprof->copied_bytes;
		}
        }
}


