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

int vftr_MPI_Scan(const void *sendbuf, void *recvbuf, int count,
                  MPI_Datatype datatype, MPI_Op op, MPI_Comm comm) {

   long long tstart = vftr_get_runtime_nsec();
   int retVal = PMPI_Scan(sendbuf, recvbuf, count, datatype, op, comm);
   long long tend = vftr_get_runtime_nsec();

   SELF_PROFILE_START_FUNCTION;
   long long t2start = tend;
   // Only intra-communicators, as the standard specifically states
   // that the scan operation is invalid for intercommunicators
   //
   // The communication pattern is the same if
   // MPI_IN_PLACE is used for the sendbuffer
   int size;
   PMPI_Comm_size(comm, &size);
   int rank;
   PMPI_Comm_rank(comm, &rank);
   if (size > 1) {
      // the precise communication pattern for a scan operation
      // strongly depends on the MPI-implementation
      // and is not specified by the standard!
      // There are two ovious possibilities
      // 1. Rank 0 sends to 1
      //    Rank 1 receives from 0 and sends to 2
      //    Rank 2 receives from 1 and sends to 3
      //    ...
      //    Rank (n-1) receives from (n-2) and sends to n
      //    Rank n receives from (n-1)
      //
      //    In this pattern every rank performs its operation and
      //    hands the result to the next rank.
      //    This pattern will be recorded here,
      //    as it has less communication overall and seems more lightweight.
      //    The actually used pattern might introduce optimizations or be a mixture of both patterns,
      //    which cannot be considered here.
      //
      // 2. Rank 0 sends to ranks 1 to n
      //    Rank 1 sends to ranks 2 to n and receives from rank 0
      //    Rank 2 sends to ranks 3 to n and receives from ranks 0 to 1
      //    Rank 3 sends to ranks 4 to n and receives from ranks 0 to 2
      //    ...
      //    Rank (n-1) sends to rank n and receives from ranks 0 to (n-2)
      //    Rank n receives from ranks 0 to (n-1)
      //
      //    This pattern will not be recorded as it involves
      //    a lot of different communications and seems to
      //    scale worse than pattern 1!
      if (rank == 0) {
         // rank 0 only sends to 1
         vftr_store_sync_message_info(send, count, datatype,
                                      rank+1, -1, comm, tstart, tend);
      } else if (rank == size - 1) {
         // the last rank will only receive from the rank before it
         vftr_store_sync_message_info(recv, count, datatype,
                                      rank-1, -1, comm, tstart, tend);
      } else {
         // all other ranks will receive from the rank before it
         // and send to the rank after it
         vftr_store_sync_message_info(recv, count, datatype,
                                      rank-1, -1, comm, tstart, tend);
         vftr_store_sync_message_info(send, count, datatype,
                                      rank+1, -1, comm, tstart, tend);
      }
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
