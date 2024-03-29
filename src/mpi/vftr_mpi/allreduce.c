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

int vftr_MPI_Allreduce(const void *sendbuf, void *recvbuf, int count,
                       MPI_Datatype datatype, MPI_Op op, MPI_Comm comm) {

   long long tstart = vftr_get_runtime_nsec();
   int retVal = PMPI_Allreduce(sendbuf, recvbuf, count, datatype, op, comm);
   long long tend = vftr_get_runtime_nsec();

   SELF_PROFILE_START_FUNCTION;
   long long t2start = tend;
   int size;
   PMPI_Comm_size(comm, &size);
   for (int i=0; i<size; i++) {
      vftr_store_sync_message_info(send, count, datatype, i, -1,
                                   comm, tstart, tend);
      // The receive is not strictly true as every process receives only one
      // data package, but due to the nature of a remote reduce
      // it is not possible to destinguish from whom.
      // There are three possibilities how to deal with this
      // 1. Don't register the receive at all
      // 2. Register the receive with count data from every remote process
      // 3. Register the receive with count/(remote size) data
      //    from every remote process
      // We selected number 2, because option 3 might not result
      // in an integer abmount of received data.
      vftr_store_sync_message_info(recv, count, datatype, i, -1,
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

int vftr_MPI_Allreduce_inplace(const void *sendbuf, void *recvbuf, int count,
                               MPI_Datatype datatype, MPI_Op op, MPI_Comm comm) {

   long long tstart = vftr_get_runtime_nsec();
   int retVal = PMPI_Allreduce(sendbuf, recvbuf, count, datatype, op, comm);
   long long tend = vftr_get_runtime_nsec();

   SELF_PROFILE_START_FUNCTION;
   long long t2start = tend;
   int size;
   PMPI_Comm_size(comm, &size);
   // For the in-place option no self communication is executed
   int rank;
   PMPI_Comm_rank(comm, &rank);

   for (int i=0; i<rank; i++) {
      vftr_store_sync_message_info(send, count, datatype, i, -1,
                                   comm, tstart, tend);
      // The receive is not strictly true as every process receives only one
      // data package, but due to the nature of a remote reduce
      // it is not possible to destinguish from whom.
      // There are three possibilities how to deal with this
      // 1. Don't register the receive at all
      // 2. Register the receive with count data from every remote process
      // 3. Register the receive with count/(remote size) data
      //    from every remote process
      // We selected number 2, because option 3 might not result
      // in an integer abmount of received data.
      vftr_store_sync_message_info(recv, count, datatype, i, -1,
                                   comm, tstart, tend);
   }
   for (int i=rank+1; i<size; i++) {
      vftr_store_sync_message_info(send, count, datatype, i, -1,
                                   comm, tstart, tend);
      // The receive is not strictly true as every process receives only one
      // data package, but due to the nature of a remote reduce
      // it is not possible to destinguish from whom.
      // There are three possibilities how to deal with this
      // 1. Don't register the receive at all
      // 2. Register the receive with count data from every remote process
      // 3. Register the receive with count/(remote size) data
      //    from every remote process
      // We selected number 2, because option 3 might not result
      // in an integer abmount of received data.
      vftr_store_sync_message_info(recv, count, datatype, i, -1,
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

int vftr_MPI_Allreduce_intercom(const void *sendbuf, void *recvbuf, int count,
                                MPI_Datatype datatype, MPI_Op op, MPI_Comm comm) {

   long long tstart = vftr_get_runtime_nsec();
   int retVal = PMPI_Allreduce(sendbuf, recvbuf, count, datatype, op, comm);
   long long tend = vftr_get_runtime_nsec();

   SELF_PROFILE_START_FUNCTION;
   long long t2start = tend;
   // Every process of group A performs the reduction within the group A
   // and stores the result on everyp process of group B and vice versa
   int size;
   PMPI_Comm_remote_size(comm, &size);
   for (int i=0; i<size; i++) {
      // translate the i-th rank in the remote group to the global rank
      int global_peer_rank = vftr_remote2global_rank(comm, i);
      // Store message info with MPI_COMM_WORLD as communicator
      // to prevent additional (and thus faulty rank translation)
      vftr_store_sync_message_info(send, count, datatype,
                                   global_peer_rank, -1, MPI_COMM_WORLD,
                                   tstart, tend);
      // The receive is not strictly true as every process receives only one
      // data package, but due to the nature of a remote reduce
      // it is not possible to destinguish from whom.
      // There are three possibilities how to deal with this
      // 1. Don't register the receive at all
      // 2. Register the receive with count data from every remote process
      // 3. Register the receive with count/(remote size) data
      //    from every remote process
      // We selected number 2, because option 3 might not result
      // in an integer abmount of received data.
      vftr_store_sync_message_info(recv, count, datatype,
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
