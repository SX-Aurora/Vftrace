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
#include "sync_messages.h"

int vftr_MPI_Recv(void *buf, int count, MPI_Datatype datatype,
                  int source, int tag, MPI_Comm comm, MPI_Status *status) {
   MPI_Status tmpstatus;
   long long tstart = vftr_get_runtime_nsec();
   int retVal = PMPI_Recv(buf, count, datatype, source, tag, comm, &tmpstatus);
   long long tend = vftr_get_runtime_nsec();

   SELF_PROFILE_START_FUNCTION;
   long long t2start = tend;
   vftr_store_sync_message_info(recv, count, datatype, tmpstatus.MPI_SOURCE,
      tmpstatus.MPI_TAG, comm, tstart, tend);

   // handle the special case of MPI_STATUS_IGNORE
   if (status != MPI_STATUS_IGNORE) {
      status->MPI_SOURCE = tmpstatus.MPI_SOURCE;
      status->MPI_TAG = tmpstatus.MPI_TAG;
      status->MPI_ERROR = tmpstatus.MPI_ERROR;
   }

   thread_t *my_thread = vftr_get_my_thread(&(vftrace.process.threadtree));
   threadstack_t *my_threadstack = vftr_get_my_threadstack(my_thread);
   vftr_stack_t *my_stack = vftrace.process.stacktree.stacks+my_threadstack->stackID;
   profile_t *my_profile = vftr_get_my_profile(my_stack, my_thread);
   long long t2end = vftr_get_runtime_nsec();

   vftr_accumulate_mpiprofiling_overhead(&(my_profile->mpiprof), t2end-t2start);

   SELF_PROFILE_END_FUNCTION;
   return retVal;
}
