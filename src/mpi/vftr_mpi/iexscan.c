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
#include "collective_requests.h"
#include "mpi_buf_addr_const.h"

int vftr_MPI_Iexscan(const void *sendbuf, void *recvbuf, int count,
                     MPI_Datatype datatype, MPI_Op op, MPI_Comm comm,
                     MPI_Request *request) {
   long long tstart = vftr_get_runtime_nsec();
   int retVal = PMPI_Iexscan(sendbuf, recvbuf, count, datatype, op, comm, request);

   SELF_PROFILE_START_FUNCTION;
   long long t2start = vftr_get_runtime_nsec();
   // Only intra-communicators, as the standard specifically states
   // that the scan operation is invalid for intercommunicators
   //
   // The communication pattern is the same if
   // MPI_IN_PLACE is used for the sendbuffer
   int size;
   PMPI_Comm_size(comm, &size);
   int rank;
   PMPI_Comm_rank(comm, &rank);
   // If the comm size is one, no communication occours
   // not even self communication
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
         int tmprank = rank+1;
         vftr_register_collective_request(send, 1, &count, &datatype,
                                          &tmprank, comm, *request,
                                          0, NULL, tstart);
      } else if (rank == size - 1) {
         // the last rank will only receive from the rank before it
         int tmprank = rank-1;
         vftr_register_collective_request(recv, 1, &count, &datatype,
                                          &tmprank, comm, *request,
                                          0, NULL, tstart);
      } else {
         // all other ranks will receive from the rank before it
         // and send to the rank after it
         int tmprank = rank-1;
         vftr_register_collective_request(recv, 1, &count, &datatype,
                                          &tmprank, comm, *request,
                                          0, NULL, tstart);
         tmprank = rank+1;
         vftr_register_collective_request(send, 1, &count, &datatype,
                                          &tmprank, comm, *request,
                                          0, NULL, tstart);
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
