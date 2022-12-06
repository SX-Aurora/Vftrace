#include "collated_stack_types.h"
#include "cudaprofiling_types.h"

// Currently, CUDA profiling is only supported for
// one MPI process and one OMP thread. Therefore, collating
// the profiles just comes down to copying the profile from
// the one stack which exists.
void vftr_collate_cudaprofiles (collated_stacktree_t *collstacktree_ptr, 
  				 stacktree_t *stacktree_ptr,
 				 int myrank, int nranks, int *nremote_profiles) {
	for (int istack = 0; istack < stacktree_ptr->nstacks; istack++) {
		stack_t *stack = stacktree_ptr->stacks + istack;
		int i_collstack = stack->gid;
		collated_stack_t *collstack = collstacktree_ptr->stacks + i_collstack;
                // CUDA profiling is only supported for one thread (i_prof = 0).
                cudaprofile_t copy_cudaprof = stack->profiling.profiles[0].cudaprof;
		cudaprofile_t *collcudaprof = &(collstack->profile.cudaprof);
		
		collcudaprof->cbid = copy_cudaprof.cbid;
                collcudaprof->n_calls = copy_cudaprof.n_calls;
                collcudaprof->t_ms = copy_cudaprof.t_ms;
                collcudaprof->memcpy_bytes[0] = copy_cudaprof.memcpy_bytes[0];
                collcudaprof->memcpy_bytes[1] = copy_cudaprof.memcpy_bytes[1];
 		collcudaprof->overhead_nsec = copy_cudaprof.overhead_nsec;
                // Start and stop events are irrelevant for the collated profile.
        }
}


