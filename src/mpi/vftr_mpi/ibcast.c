#include <mpi.h>

#include <stdlib.h>

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
#include "rank_translate.h"

int vftr_MPI_Ibcast(void *buffer, int count, MPI_Datatype datatype,
                   int root, MPI_Comm comm, MPI_Request *request) {
   long long tstart = vftr_get_runtime_nsec();
   int retVal = PMPI_Ibcast(buffer, count, datatype, root, comm, request);
   SELF_PROFILE_START_FUNCTION;
   long long t2start = vftr_get_runtime_nsec();
   // in intracommunicators the expected behaviour is to
   // gather from root to all other processes in the communicator
   int rank;
   PMPI_Comm_rank(comm, &rank);
   if (rank == root) {
      int size;
      PMPI_Comm_size(comm, &size);
      // allocate memory for the temporary arrays
      // to register communication request
      int *tmpcount = (int*) malloc(sizeof(int)*size);
      MPI_Datatype *tmpdatatype = (MPI_Datatype*) malloc(sizeof(MPI_Datatype)*size);
      int *tmppeer_ranks = (int*) malloc(sizeof(int)*size);
      // messages to be send
      for (int i=0; i<size; i++) {
         tmpcount[i] = count;
         tmpdatatype[i] = datatype;
         tmppeer_ranks[i] = i;
      }
      vftr_register_collective_request(send, size, tmpcount, tmpdatatype,
                                       tmppeer_ranks, comm,
                                       *request, 0, NULL, tstart);
      // cleanup temporary arrays
      free(tmpcount);
      tmpcount = NULL;
      free(tmpdatatype);
      tmpdatatype = NULL;
      free(tmppeer_ranks);
      tmppeer_ranks = NULL;
   } else {
      vftr_register_collective_request(recv, 1, &count, &datatype, &root,
                                       comm, *request, 0, NULL, tstart);
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

int vftr_MPI_Ibcast_intercom(void *buffer, int count, MPI_Datatype datatype,
                             int root, MPI_Comm comm, MPI_Request *request) {
   long long tstart = vftr_get_runtime_nsec();
   int retVal = PMPI_Ibcast(buffer, count, datatype, root, comm, request);
   SELF_PROFILE_START_FUNCTION;
   long long t2start = vftr_get_runtime_nsec();
   // in intercommunicators the behaviour is more complicated
   // There are two groups A and B
   // In group A the root process is located.
   if (root == MPI_ROOT) {
      // The root process get the special process wildcard MPI_ROOT
      // get the size of group B
      int size;
      PMPI_Comm_remote_size(comm, &size);
      int *tmpcount = (int*) malloc(sizeof(int)*size);
      MPI_Datatype *tmpdatatype = (MPI_Datatype*) malloc(sizeof(MPI_Datatype)*size);
      int *tmppeer_ranks= (int*) malloc(sizeof(int)*size);
      // messages to be send
      for (int i=0; i<size; i++) {
         tmpcount[i] = count;
         tmpdatatype[i] = datatype;
         // translate the i-th rank in the remote group to the global rank
         tmppeer_ranks[i] = vftr_remote2global_rank(comm, i);
      }
      // Register request with MPI_COMM_WORLD as communicator
      // to prevent additional (and thus faulty rank translation)
      vftr_register_collective_request(send, size, tmpcount, tmpdatatype,
                                       tmppeer_ranks, MPI_COMM_WORLD,
                                       *request, 0, NULL, tstart);
      // cleanup temporary arrays
      free(tmpcount);
      tmpcount = NULL;
      free(tmpdatatype);
      tmpdatatype = NULL;
      free(tmppeer_ranks);
      tmppeer_ranks = NULL;
   } else if (root == MPI_PROC_NULL) {
      // All other processes from group A pass wildcard MPI_PROC NULL
      // They do not participate in intercommunicator gather
      ;
   } else {
      // All other processes must be located in group B
      // root is the rank-id in group A Therefore no problems with
      // rank translation should arise
      vftr_register_collective_request(recv, 1, &count, &datatype, &root,
                                       comm, *request, 0, NULL, tstart);
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
