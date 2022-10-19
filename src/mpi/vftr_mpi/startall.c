#include <mpi.h>

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

int vftr_MPI_Startall(int count, MPI_Request *array_of_requests) {
   long long tstart = vftr_get_runtime_nsec();
   int retVal = PMPI_Startall(count, array_of_requests);

   SELF_PROFILE_START_FUNCTION;
   long long t2start = vftr_get_runtime_nsec();
   for (int ireq=0; ireq<count; ireq++) {
      vftr_activate_pers_request(array_of_requests[ireq], tstart) ;
   }

   thread_t *my_thread = vftr_get_my_thread(&(vftrace.process.threadtree));
   threadstack_t *my_threadstack = vftr_get_my_threadstack(my_thread);
   stack_t *my_stack = vftrace.process.stacktree.stacks+my_threadstack->stackID;
   profile_t *my_profile = vftr_get_my_profile(my_stack, my_thread);
   long long t2end = vftr_get_runtime_nsec();

   vftr_accumulate_mpiprofiling_overhead(&(my_profile->mpiprof), t2end-t2start);

   SELF_PROFILE_END_FUNCTION;
   return retVal;
}
