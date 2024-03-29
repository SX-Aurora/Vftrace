#include <mpi.h>

#include "self_profile.h"
#include "rank_translate.h"
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

int vftr_MPI_Alltoall(const void *sendbuf, int sendcount,
                      MPI_Datatype sendtype, void *recvbuf,
                      int recvcount, MPI_Datatype recvtype,
                      MPI_Comm comm) {
   long long tstart = vftr_get_runtime_nsec();
   int retVal = PMPI_Alltoall(sendbuf, sendcount, sendtype, recvbuf,
                              recvcount, recvtype, comm);
   long long tend = vftr_get_runtime_nsec();

   SELF_PROFILE_START_FUNCTION;
   long long t2start = tend;
   int size;
   PMPI_Comm_size(comm, &size);
   for (int i=0; i<size; i++) {
      vftr_store_sync_message_info(send, sendcount, sendtype, i, 0,
                                   comm, tstart, tend);
      vftr_store_sync_message_info(recv, recvcount, recvtype, i, 0,
                                   comm, tstart, tend);
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

int vftr_MPI_Alltoall_inplace(const void *sendbuf, int sendcount,
                              MPI_Datatype sendtype, void *recvbuf,
                              int recvcount, MPI_Datatype recvtype,
                              MPI_Comm comm) {
   long long tstart = vftr_get_runtime_nsec();
   int retVal = PMPI_Alltoall(sendbuf, sendcount, sendtype, recvbuf,
                              recvcount, recvtype, comm);
   long long tend = vftr_get_runtime_nsec();

   SELF_PROFILE_START_FUNCTION;
   long long t2start = tend;
   int size;
   PMPI_Comm_size(comm, &size);
   sendcount = recvcount;
   sendtype = recvtype;
   // For the in-place option no self communication is executed
   int rank;
   PMPI_Comm_rank(comm, &rank);
   for (int i=0; i<rank; i++) {
      vftr_store_sync_message_info(send, sendcount, sendtype, i, 0,
                                   comm, tstart, tend);
      vftr_store_sync_message_info(recv, recvcount, recvtype, i, 0,
                                   comm, tstart, tend);
   }
   for (int i=rank+1; i<size; i++) {
      vftr_store_sync_message_info(send, sendcount, sendtype, i, 0,
                                   comm, tstart, tend);
      vftr_store_sync_message_info(recv, recvcount, recvtype, i, 0,
                                   comm, tstart, tend);
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

int vftr_MPI_Alltoall_intercom(const void *sendbuf, int sendcount,
                               MPI_Datatype sendtype, void *recvbuf,
                               int recvcount, MPI_Datatype recvtype,
                               MPI_Comm comm) {
   long long tstart = vftr_get_runtime_nsec();
   int retVal = PMPI_Alltoall(sendbuf, sendcount, sendtype, recvbuf,
                              recvcount, recvtype, comm);
   long long tend = vftr_get_runtime_nsec();

   SELF_PROFILE_START_FUNCTION;
   long long t2start = tend;
   // Every process of group A sends sendcount sendtypes
   // to and receives recvcount recvtypes from
   // every process in group Band vice versa.
   int size;
   PMPI_Comm_remote_size(comm, &size);
   for (int i=0; i<size; i++) {
      // translate the i-th rank in the remote group to the global rank
      int global_peer_rank = vftr_remote2global_rank(comm, i);
      // Store message info with MPI_COMM_WORLD as communicator
      // to prevent additional (and thus faulty rank translation)
      vftr_store_sync_message_info(send, sendcount, sendtype,
                                   global_peer_rank, -1, MPI_COMM_WORLD,
                                   tstart, tend);
      vftr_store_sync_message_info(recv, recvcount, recvtype,
                                   global_peer_rank, -1, MPI_COMM_WORLD,
                                   tstart, tend);
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
