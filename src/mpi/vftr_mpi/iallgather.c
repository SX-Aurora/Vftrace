#include <mpi.h>

#include <stdlib.h>

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
#include "collective_requests.h"

int vftr_MPI_Iallgather(const void *sendbuf, int sendcount,
                        MPI_Datatype sendtype, void *recvbuf,
                        int recvcount, MPI_Datatype recvtype,
                        MPI_Comm comm, MPI_Request *request) {
   long long tstart = vftr_get_runtime_nsec();
   int retVal = PMPI_Iallgather(sendbuf, sendcount, sendtype, recvbuf,
                                recvcount, recvtype, comm, request);

   SELF_PROFILE_START_FUNCTION;
   long long t2start = vftr_get_runtime_nsec();
   int size;
   PMPI_Comm_size(comm, &size);
   // allocate memory for the temporary arrays
   // to register communication request
   int *tmpcount = (int*) malloc(sizeof(int)*size);
   MPI_Datatype *tmptype = (MPI_Datatype*) malloc(sizeof(MPI_Datatype)*size);
   int *peer_ranks = (int*) malloc(sizeof(int)*size);
   // messages to be send
   for (int i=0; i<size; i++) {
      tmpcount[i] = sendcount;
      tmptype[i] = sendtype;
      peer_ranks[i] = i;
   }
   vftr_register_collective_request(send, size, tmpcount, tmptype, peer_ranks,
                                    comm, *request, 0, NULL, tstart);
   // messages to be received
   for (int i=0; i<size; i++) {
      tmpcount[i] = recvcount;
      tmptype[i] = recvtype;
      peer_ranks[i] = i;
   }
   vftr_register_collective_request(recv, size, tmpcount, tmptype, peer_ranks,
                                    comm, *request, 0, NULL, tstart);
   // cleanup temporary arrays
   free(tmpcount);
   tmpcount = NULL;
   free(tmptype);
   tmptype = NULL;
   free(peer_ranks);
   peer_ranks = NULL;

   thread_t *my_thread = vftr_get_my_thread(&(vftrace.process.threadtree));
   threadstack_t *my_threadstack = vftr_get_my_threadstack(my_thread);
   vftr_stack_t *my_stack = vftrace.process.stacktree.stacks+my_threadstack->stackID;
   profile_t *my_profile = vftr_get_my_profile(my_stack, my_thread);
   long long t2end = vftr_get_runtime_nsec();

   vftr_accumulate_mpiprofiling_overhead(&(my_profile->mpiprof), t2end-t2start);

   SELF_PROFILE_END_FUNCTION;
   return retVal;
}

int vftr_MPI_Iallgather_inplace(const void *sendbuf, int sendcount,
                                MPI_Datatype sendtype, void *recvbuf,
                                int recvcount, MPI_Datatype recvtype,
                                MPI_Comm comm, MPI_Request *request) {
   long long tstart = vftr_get_runtime_nsec();
   int retVal = PMPI_Iallgather(sendbuf, sendcount, sendtype, recvbuf,
                                recvcount, recvtype, comm, request);

   SELF_PROFILE_START_FUNCTION;
   long long t2start = vftr_get_runtime_nsec();
   int size;
   PMPI_Comm_size(comm, &size);
   if (size > 1) {
      // For the in-place option no self communication is executed
      int rank;
      PMPI_Comm_rank(comm, &rank);

      // allocate memory for the temporary arrays
      // to register communication request
      int *tmpcount = (int*) malloc(sizeof(int)*(size-1));
      MPI_Datatype *tmptype = (MPI_Datatype*)
                              malloc(sizeof(MPI_Datatype)*(size-1));
      int *peer_ranks = (int*) malloc(sizeof(int)*(size-1));
      // messages to be send and received
      int idx = 0;
      for (int i=0; i<rank; i++) {
         tmpcount[idx] = recvcount;
         tmptype[idx] = recvtype;
         peer_ranks[idx] = i;
         idx++;
      }
      for (int i=rank+1; i<size; i++) {
         tmpcount[idx] = recvcount;
         tmptype[idx] = recvtype;
         peer_ranks[idx] = i;
         idx++;
      }
      // sendcount and sendtype are ignored,
      // thus the tmpcount and tmptype array is identical
      vftr_register_collective_request(send, size-1, tmpcount, tmptype, peer_ranks,
                                       comm, *request, 0, NULL, tstart);
      vftr_register_collective_request(recv, size-1, tmpcount, tmptype, peer_ranks,
                                       comm, *request, 0, NULL, tstart);
      // cleanup temporary arrays
      free(tmpcount);
      tmpcount = NULL;
      free(tmptype);
      tmptype = NULL;
      free(peer_ranks);
      peer_ranks = NULL;
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

int vftr_MPI_Iallgather_intercom(const void *sendbuf, int sendcount,
                                 MPI_Datatype sendtype, void *recvbuf,
                                 int recvcount, MPI_Datatype recvtype,
                                 MPI_Comm comm, MPI_Request *request) {
   long long tstart = vftr_get_runtime_nsec();
   int retVal = PMPI_Iallgather(sendbuf, sendcount, sendtype, recvbuf,
                                recvcount, recvtype, comm, request);

   SELF_PROFILE_START_FUNCTION;
   long long t2start = vftr_get_runtime_nsec();
   // Every process of group A sends sendcount data to and
   // receives recvcount data from every process in group B and
   // vice versa
   int size;
   PMPI_Comm_remote_size(comm, &size);
   // allocate memory for the temporary arrays
   // to register communication request
   int *tmpcount = (int*) malloc(sizeof(int)*size);
   MPI_Datatype *tmptype = (MPI_Datatype*) malloc(sizeof(MPI_Datatype)*size);
   int *peer_ranks = (int*) malloc(sizeof(int)*size);
   // messages to be send
   for (int i=0; i<size; i++) {
      tmpcount[i] = sendcount;
      tmptype[i] = sendtype;
      // translate the i-th rank in the remote group to the global rank
      peer_ranks[i] = vftr_remote2global_rank(comm, i);
   }
   // Register request with MPI_COMM_WORLD as communicator
   // to prevent additional (and thus faulty rank translation)
   vftr_register_collective_request(send, size, tmpcount, tmptype, peer_ranks,
                                    MPI_COMM_WORLD, *request, 0, NULL, tstart);
   // messages to be received
   for (int i=0; i<size; i++) {
      tmpcount[i] = recvcount;
      tmptype[i] = recvtype;
      // translate the i-th rank in the remote group to the global rank
      peer_ranks[i] = vftr_remote2global_rank(comm, i);
   }
   // Register request with MPI_COMM_WORLD as communicator
   // to prevent additional (and thus faulty rank translation)
   vftr_register_collective_request(recv, size, tmpcount, tmptype, peer_ranks,
                                    MPI_COMM_WORLD, *request, 0, NULL, tstart);
   // cleanup temporary arrays
   free(tmpcount);
   tmpcount = NULL;
   free(tmptype);
   tmptype = NULL;
   free(peer_ranks);
   peer_ranks = NULL;

   thread_t *my_thread = vftr_get_my_thread(&(vftrace.process.threadtree));
   threadstack_t *my_threadstack = vftr_get_my_threadstack(my_thread);
   vftr_stack_t *my_stack = vftrace.process.stacktree.stacks+my_threadstack->stackID;
   profile_t *my_profile = vftr_get_my_profile(my_stack, my_thread);
   long long t2end = vftr_get_runtime_nsec();

   vftr_accumulate_mpiprofiling_overhead(&(my_profile->mpiprof), t2end-t2start);

   SELF_PROFILE_END_FUNCTION;
   return retVal;
}
