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

int vftr_MPI_Igather(const void *sendbuf, int sendcount,
                     MPI_Datatype sendtype, void *recvbuf, int recvcount,
                     MPI_Datatype recvtype, int root, MPI_Comm comm,
                     MPI_Request *request) {
   long long tstart = vftr_get_runtime_nsec();
   int retVal = PMPI_Igather(sendbuf, sendcount, sendtype, recvbuf, recvcount,
                            recvtype, root, comm, request);

   SELF_PROFILE_START_FUNCTION;
   long long t2start = vftr_get_runtime_nsec();
   // determine if inter or intra communicator
   int isintercom;
   PMPI_Comm_test_inter(comm, &isintercom);
   // in intracommunicators the expected behaviour is to
   // bcast from root to all other processes in the communicator
   int rank;
   PMPI_Comm_rank(comm, &rank);
   if (rank == root) {
      int size;
      PMPI_Comm_size(comm, &size);
      // self communication of root process
      vftr_register_collective_request(send, 1, &sendcount, &sendtype, &root,
                                       comm, *request, 0, NULL, tstart);
      // allocate memory for the temporary arrays
      // to register communication request
      int *tmprecvcount = (int*) malloc(sizeof(int)*size);
      MPI_Datatype *tmprecvtype = (MPI_Datatype*) malloc(sizeof(MPI_Datatype)*size);
      int *tmppeer_ranks= (int*) malloc(sizeof(int)*size);
      // messeges to be received
      for (int i=0; i<size; i++) {
         tmprecvcount[i] = recvcount;
         tmprecvtype[i] = recvtype;
         tmppeer_ranks[i] = i;
      }
      vftr_register_collective_request(recv, size, tmprecvcount, tmprecvtype,
                                       tmppeer_ranks, comm, *request,
                                       0, NULL, tstart);
      // cleanup temporary arrays
      free(tmprecvcount);
      tmprecvcount = NULL;
      free(tmprecvtype);
      tmprecvtype = NULL;
      free(tmppeer_ranks);
      tmppeer_ranks = NULL;

   } else {
      vftr_register_collective_request(send, 1, &sendcount, &sendtype, &root,
                                       comm, *request, 0, NULL, tstart);
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

int vftr_MPI_Igather_inplace(const void *sendbuf, int sendcount,
                             MPI_Datatype sendtype, void *recvbuf,
                             int recvcount, MPI_Datatype recvtype,
                             int root, MPI_Comm comm,
                             MPI_Request *request) {
   long long tstart = vftr_get_runtime_nsec();
   int retVal = PMPI_Igather(sendbuf, sendcount, sendtype, recvbuf, recvcount,
                            recvtype, root, comm, request);
   SELF_PROFILE_START_FUNCTION;
   long long t2start = vftr_get_runtime_nsec();
   // determine if inter or intra communicator
   int isintercom;
   PMPI_Comm_test_inter(comm, &isintercom);
   // in intracommunicators the expected behaviour is to
   // bcast from root to all other processes in the communicator
   int rank;
   PMPI_Comm_rank(comm, &rank);
   if (rank == root) {
      int size;
      PMPI_Comm_size(comm, &size);
      if (size > 1) {
         // For the in-place option no self communication is executed
         // allocate memory for the temporary arrays
         // to register communication request
         int *tmprecvcount = (int*) malloc(sizeof(int)*(size-1));
         MPI_Datatype *tmprecvtype = (MPI_Datatype*) malloc(sizeof(MPI_Datatype)*(size-1));
         int *tmppeer_ranks= (int*) malloc(sizeof(int)*(size-1));
         // messeges to be received
         int idx = 0;
         for (int i=0; i<rank; i++) {
            tmprecvcount[idx] = recvcount;
            tmprecvtype[idx] = recvtype;
            tmppeer_ranks[idx] = i;
            idx++;
         }
         for (int i=rank+1; i<size; i++) {
            tmprecvcount[idx] = recvcount;
            tmprecvtype[idx] = recvtype;
            tmppeer_ranks[idx] = i;
            idx++;
         }
         vftr_register_collective_request(recv, size-1, tmprecvcount, tmprecvtype,
                                          tmppeer_ranks, comm, *request,
                                          0, NULL, tstart);
         // cleanup temporary arrays
         free(tmprecvcount);
         tmprecvcount = NULL;
         free(tmprecvtype);
         tmprecvtype = NULL;
         free(tmppeer_ranks);
         tmppeer_ranks = NULL;
      }
   } else {
      vftr_register_collective_request(send, 1, &sendcount, &sendtype, &root,
                                       comm, *request, 0, NULL, tstart);
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

int vftr_MPI_Igather_intercom(const void *sendbuf, int sendcount,
                              MPI_Datatype sendtype, void *recvbuf,
                              int recvcount, MPI_Datatype recvtype,
                              int root, MPI_Comm comm,
                              MPI_Request *request) {
   long long tstart = vftr_get_runtime_nsec();
   int retVal = PMPI_Igather(sendbuf, sendcount, sendtype, recvbuf, recvcount,
                            recvtype, root, comm, request);

   SELF_PROFILE_START_FUNCTION;
   long long t2start = vftr_get_runtime_nsec();
   // determine if inter or intra communicator
   int isintercom;
   PMPI_Comm_test_inter(comm, &isintercom);
   // in intercommunicators the behaviour is more complicated
   // There are two groups A and B
   // In group A the root process is located.
   if (root == MPI_ROOT) {
      // The root process get the special process wildcard MPI_ROOT
      // get the size of group B
      int size;
      PMPI_Comm_remote_size(comm, &size);
      int *tmprecvcount = (int*) malloc(sizeof(int)*size);
      MPI_Datatype *tmprecvtype = (MPI_Datatype*) malloc(sizeof(MPI_Datatype)*size);
      int *tmppeer_ranks= (int*) malloc(sizeof(int)*size);
      // messages to be received
      for (int i=0; i<size; i++) {
         tmprecvcount[i] = recvcount;
         tmprecvtype[i] = recvtype;
         // translate the i-th rank in group B to the global rank
         tmppeer_ranks[i] = vftr_remote2global_rank(comm, i);
      }
      // Register request with MPI_COMM_WORLD as communicator
      // to prevent additional (and thus faulty rank translation)
      vftr_register_collective_request(recv, size, tmprecvcount, tmprecvtype,
                                       tmppeer_ranks, MPI_COMM_WORLD,
                                       *request, 0, NULL, tstart);
      // cleanup temporary arrays
      free(tmprecvcount);
      tmprecvcount = NULL;
      free(tmprecvtype);
      tmprecvtype = NULL;
      free(tmppeer_ranks);
      tmppeer_ranks = NULL;
   } else if (root == MPI_PROC_NULL) {
      // All other processes from group A pass wildcard MPI_PROC NULL
      // They do not participate in intercommunicator bcasts
      ;
   } else {
      // All other processes must be located in group B
      // root is the rank-id in group A Therefore no problems with
      // rank translation should arise
      vftr_register_collective_request(send, 1, &sendcount, &sendtype, &root,
                                       comm, *request, 0, NULL, tstart);
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
