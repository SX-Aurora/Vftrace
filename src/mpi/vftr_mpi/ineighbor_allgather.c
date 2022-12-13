#include <stdlib.h>

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
#include "collective_requests.h"
#include "cart_comms.h"

int vftr_MPI_Ineighbor_allgather_graph(const void *sendbuf, int sendcount,
                                       MPI_Datatype sendtype, void *recvbuf,
                                       int recvcount, MPI_Datatype recvtype,
                                       MPI_Comm comm, MPI_Request *request) {
   long long tstart = vftr_get_runtime_nsec();
   int retVal = PMPI_Ineighbor_allgather(sendbuf, sendcount, sendtype, recvbuf,
                                         recvcount, recvtype, comm, request);

   SELF_PROFILE_START_FUNCTION;
   long long t2start = vftr_get_runtime_nsec();
   int rank;
   PMPI_Comm_rank(comm, &rank);
   int nneighbors;
   PMPI_Graph_neighbors_count(comm, rank, &nneighbors);
   int *neighbors = (int*) malloc(nneighbors*sizeof(int));
   PMPI_Graph_neighbors(comm, rank, nneighbors, neighbors);
   // allocate memory for the temporary arrays
   // to register communication request
   int *tmpcount = (int*) malloc(sizeof(int)*nneighbors);
   MPI_Datatype *tmptype = (MPI_Datatype*) malloc(sizeof(MPI_Datatype)*nneighbors);
   // messages to be send
   for (int ineighbor=0; ineighbor<nneighbors; ineighbor++) {
      tmpcount[ineighbor] = sendcount;
      tmptype[ineighbor] = sendtype;
   }
   vftr_register_collective_request(send, nneighbors, tmpcount, tmptype, neighbors,
                                    comm, *request, 0, NULL, tstart);
   // messages to be send
   for (int ineighbor=0; ineighbor<nneighbors; ineighbor++) {
      tmpcount[ineighbor] = recvcount;
      tmptype[ineighbor] = recvtype;
   }
   vftr_register_collective_request(recv, nneighbors, tmpcount, tmptype, neighbors,
                                    comm, *request, 0, NULL, tstart);
   // cleanup temporary arrays
   free(tmpcount);
   tmpcount = NULL;
   free(tmptype);
   tmptype = NULL;
   free(neighbors);
   neighbors = NULL;

   thread_t *my_thread = vftr_get_my_thread(&(vftrace.process.threadtree));
   threadstack_t *my_threadstack = vftr_get_my_threadstack(my_thread);
   vftr_stack_t *my_stack = vftrace.process.stacktree.stacks+my_threadstack->stackID;
   profile_t *my_profile = vftr_get_my_profile(my_stack, my_thread);
   long long t2end = vftr_get_runtime_nsec();

   vftr_accumulate_mpiprofiling_overhead(&(my_profile->mpiprof), t2end-t2start);

   SELF_PROFILE_END_FUNCTION;
   return retVal;
}

int vftr_MPI_Ineighbor_allgather_cart(const void *sendbuf, int sendcount,
                                      MPI_Datatype sendtype, void *recvbuf,
                                      int recvcount, MPI_Datatype recvtype,
                                      MPI_Comm comm, MPI_Request *request) {
   long long tstart = vftr_get_runtime_nsec();
   int retVal = PMPI_Ineighbor_allgather(sendbuf, sendcount, sendtype, recvbuf,
                                         recvcount, recvtype, comm, request);

   SELF_PROFILE_START_FUNCTION;
   long long t2start = vftr_get_runtime_nsec();
   int rank;
   PMPI_Comm_rank(comm, &rank);
   int nneighbors;
   int *neighbors;
   vftr_mpi_cart_neighbor_ranks(comm, &nneighbors, &neighbors);
   // allocate memory for the temporary arrays
   // to register communication request
   int *tmpcount = (int*) malloc(sizeof(int)*nneighbors);
   MPI_Datatype *tmptype = (MPI_Datatype*) malloc(sizeof(MPI_Datatype)*nneighbors);
   // messages to be send
   for (int ineighbor=0; ineighbor<nneighbors; ineighbor++) {
      tmpcount[ineighbor] = sendcount;
      tmptype[ineighbor] = sendtype;
   }
   vftr_register_collective_request(send, nneighbors, tmpcount, tmptype, neighbors,
                                    comm, *request, 0, NULL, tstart);
   // messages to be received
   for (int ineighbor=0; ineighbor<nneighbors; ineighbor++) {
      tmpcount[ineighbor] = recvcount;
      tmptype[ineighbor] = recvtype;
   }
   vftr_register_collective_request(recv, nneighbors, tmpcount, tmptype, neighbors,
                                    comm, *request, 0, NULL, tstart);
   // cleanup temporary arrays
   free(tmpcount);
   tmpcount = NULL;
   free(tmptype);
   tmptype = NULL;
   free(neighbors);
   neighbors = NULL;

   thread_t *my_thread = vftr_get_my_thread(&(vftrace.process.threadtree));
   threadstack_t *my_threadstack = vftr_get_my_threadstack(my_thread);
   vftr_stack_t *my_stack = vftrace.process.stacktree.stacks+my_threadstack->stackID;
   profile_t *my_profile = vftr_get_my_profile(my_stack, my_thread);
   long long t2end = vftr_get_runtime_nsec();

   vftr_accumulate_mpiprofiling_overhead(&(my_profile->mpiprof), t2end-t2start);

   SELF_PROFILE_END_FUNCTION;
   return retVal;
}

int vftr_MPI_Ineighbor_allgather_dist_graph(const void *sendbuf, int sendcount,
                                            MPI_Datatype sendtype, void *recvbuf,
                                            int recvcount, MPI_Datatype recvtype,
                                            MPI_Comm comm, MPI_Request *request) {
   long long tstart = vftr_get_runtime_nsec();
   int retVal = PMPI_Ineighbor_allgather(sendbuf, sendcount, sendtype, recvbuf,
                                         recvcount, recvtype, comm, request);

   SELF_PROFILE_START_FUNCTION;
   long long t2start = vftr_get_runtime_nsec();
   // first obtain the distributed graph info for this process
   int ninneighbors;
   int noutneighbors;
   int weighted;
   PMPI_Dist_graph_neighbors_count(comm, &ninneighbors,
                                   &noutneighbors, &weighted);
   int *inneighbors = (int*) malloc(ninneighbors*sizeof(int));
   int *inweights = (int*) malloc(ninneighbors*sizeof(int));
   int *outneighbors = (int*) malloc(noutneighbors*sizeof(int));
   int *outweights = (int*) malloc(noutneighbors*sizeof(int));
   PMPI_Dist_graph_neighbors(comm,
                             ninneighbors, inneighbors, inweights,
                             noutneighbors, outneighbors, outweights);
   // select the bigger number of neigbours to only allocate once
   int size = (ninneighbors > noutneighbors) ? ninneighbors : noutneighbors;
   // allocate memory for the temporary arrays
   // to register communication request
   int *tmpcount = (int*) malloc(sizeof(int)*size);
   MPI_Datatype *tmptype = (MPI_Datatype*) malloc(sizeof(MPI_Datatype)*size);
   // messages to be send
   for (int ineighbor=0; ineighbor<noutneighbors; ineighbor++) {
      tmpcount[ineighbor] = sendcount;
      tmptype[ineighbor] = sendtype;
   }
   vftr_register_collective_request(send, noutneighbors, tmpcount, tmptype, outneighbors,
                                    comm, *request, 0, NULL, tstart);

   // messages to be received
   for (int ineighbor=0; ineighbor<ninneighbors; ineighbor++) {
      tmpcount[ineighbor] = recvcount;
      tmptype[ineighbor] = recvtype;
   }
   vftr_register_collective_request(recv, ninneighbors, tmpcount, tmptype, inneighbors,
                                    comm, *request, 0, NULL, tstart);
   // cleanup temporary arrays
   free(tmpcount);
   tmpcount = NULL;
   free(tmptype);
   tmptype = NULL;
   free(inneighbors);
   inneighbors = NULL;
   free(inweights);
   inweights = NULL;
   free(outneighbors);
   outneighbors = NULL;
   free(outweights);
   outweights = NULL;


   thread_t *my_thread = vftr_get_my_thread(&(vftrace.process.threadtree));
   threadstack_t *my_threadstack = vftr_get_my_threadstack(my_thread);
   vftr_stack_t *my_stack = vftrace.process.stacktree.stacks+my_threadstack->stackID;
   profile_t *my_profile = vftr_get_my_profile(my_stack, my_thread);
   long long t2end = vftr_get_runtime_nsec();

   vftr_accumulate_mpiprofiling_overhead(&(my_profile->mpiprof), t2end-t2start);

   SELF_PROFILE_END_FUNCTION;
   return retVal;
}
