#include <mpi.h>

#include <stdbool.h>

#include "self_profile.h"
#include "thread_types.h"
#include "threads.h"
#include "threadstack_types.h"
#include "threadstacks.h"
#include "stack_types.h"
#include "stacks.h"
#include "profiling_types.h"
#include "profiling.h"
#include "mpiprofiling.h"
#include "timer.h"
#include "requests.h"
#include "p2p_requests.h"
#include "requests.h"

int vftr_MPI_Request_free(MPI_Request *request) {
   SELF_PROFILE_START_FUNCTION;
   long long t2start = vftr_get_runtime_nsec();
   vftr_request_t *matched_request = vftr_search_request(*request);
   if (matched_request != NULL) {
         matched_request->marked_for_deallocation = true;
   } else {
      PMPI_Request_free(request);
   }
   *request = MPI_REQUEST_NULL;

   thread_t *my_thread = vftr_get_my_thread(&(vftrace.process.threadtree));
   threadstack_t *my_threadstack = vftr_get_my_threadstack(my_thread);
   vftr_stack_t *my_stack = vftrace.process.stacktree.stacks+my_threadstack->stackID;
   profile_t *my_profile = vftr_get_my_profile(my_stack, my_thread);
   long long t2end = vftr_get_runtime_nsec();

   vftr_accumulate_mpiprofiling_overhead(&(my_profile->mpiprof), t2end-t2start);

   int retVal = 0;
   SELF_PROFILE_END_FUNCTION;
   return retVal;
}
