#include <string.h>

#include "vftrace_state.h"
#include "collated_stack_types.h"
#include "papiprofiling_types.h"

void vftr_collate_papiprofiles (collated_stacktree_t *collstacktree_ptr,
			        stacktree_t *stacktree_ptr,
				int myrank, int nranks, int *nremote_profiles) {
   for (int istack = 0; istack < stacktree_ptr->nstacks; istack++) {
      vftr_stack_t *stack = stacktree_ptr->stacks + istack;
      int i_collstack = stack->gid;
      collated_stack_t *collstack = collstacktree_ptr->stacks + i_collstack;
      papiprofile_t copy_papiprof = stack->profiling.profiles[0].papiprof;
      papiprofile_t *collpapiprof = &(collstack->profile.papiprof);
     
      collpapiprof->counters = (long long*)malloc (vftrace.papi_state.n_available_events * sizeof(long long));
      memcpy (collpapiprof->counters, copy_papiprof.counters, vftrace.papi_state.n_available_events * sizeof(long long));
   }
}
