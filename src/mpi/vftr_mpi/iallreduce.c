/*
   This file is part of Vftrace.

   Vftrace is free software; you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation; either version 2 of the License, or
   (at your option) any later version.

   Vftrace is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License along
   with this program; if not, write to the Free Software Foundation, Inc.,
   51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
*/

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

int vftr_MPI_Iallreduce(const void *sendbuf, void *recvbuf, int count,
                        MPI_Datatype datatype, MPI_Op op, MPI_Comm comm,
                        MPI_Request *request) {

   long long tstart = vftr_get_runtime_usec();
   int retVal = PMPI_Iallreduce(sendbuf, recvbuf, count, datatype, op, comm, request);
   long long tend = vftr_get_runtime_usec();

   SELF_PROFILE_START_FUNCTION;
   long long t2start = tend;
   int size;
   PMPI_Comm_size(comm, &size);
   // allocate memory for the temporary arrays
   // to register communication request
   int *tmpcount = (int*) malloc(sizeof(int)*size);
   MPI_Datatype *tmptype = (MPI_Datatype*) malloc(sizeof(MPI_Datatype)*size);
   int *peer_ranks = (int*) malloc(sizeof(int)*size);
   // messages to be send
   for (int i=0; i<size; i++) {
      tmpcount[i] = count;
      tmptype[i] = datatype;
      peer_ranks[i] = i;
   }
   vftr_register_collective_request(send, size, tmpcount, tmptype, peer_ranks,
                                    comm, *request, 0, NULL, tstart);
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
   stack_t *my_stack = vftrace.process.stacktree.stacks+my_threadstack->stackID;
   profile_t *my_profile = vftr_get_my_profile(my_stack, my_thread);
   long long t2end = vftr_get_runtime_usec();

   vftr_accumulate_mpiprofiling_overhead(&(my_profile->mpiProf), t2end-t2start);

   SELF_PROFILE_END_FUNCTION;
   return retVal;
}

int vftr_MPI_Iallreduce_inplace(const void *sendbuf, void *recvbuf, int count,
                                MPI_Datatype datatype, MPI_Op op, MPI_Comm comm,
                                MPI_Request *request) {

   long long tstart = vftr_get_runtime_usec();
   int retVal = PMPI_Iallreduce(sendbuf, recvbuf, count, datatype, op, comm, request);
   long long tend = vftr_get_runtime_usec();

   SELF_PROFILE_START_FUNCTION;
   long long t2start = tend;
   int size;
   PMPI_Comm_size(comm, &size);
   if (size > 1) {
      // For the in-place option no self communication is executed
      int rank;
      PMPI_Comm_rank(comm, &rank);

      // allocate memory for the temporary arrays
      // to register communication request
      int *tmpcount = (int*) malloc(sizeof(int)*(size-1));
      MPI_Datatype *tmptype = (MPI_Datatype*) malloc(sizeof(MPI_Datatype)*(size-1));
      int *peer_ranks = (int*) malloc(sizeof(int)*(size-1));
      // messages to be send
      int idx = 0;
      for (int i=0; i<rank; i++) {
         tmpcount[idx] = count;
         tmptype[idx] = datatype;
         peer_ranks[idx] = i;
         idx++;
      }
      for (int i=rank+1; i<size; i++) {
         tmpcount[idx] = count;
         tmptype[idx] = datatype;
         peer_ranks[idx] = i;
         idx++;
      }
      vftr_register_collective_request(send, size-1, tmpcount, tmptype, peer_ranks,
                                       comm, *request, 0, NULL, tstart);
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
   stack_t *my_stack = vftrace.process.stacktree.stacks+my_threadstack->stackID;
   profile_t *my_profile = vftr_get_my_profile(my_stack, my_thread);
   long long t2end = vftr_get_runtime_usec();

   vftr_accumulate_mpiprofiling_overhead(&(my_profile->mpiProf), t2end-t2start);

   SELF_PROFILE_END_FUNCTION;
   return retVal;
}

int vftr_MPI_Iallreduce_intercom(const void *sendbuf, void *recvbuf, int count,
                                 MPI_Datatype datatype, MPI_Op op, MPI_Comm comm,
                                 MPI_Request *request) {

   long long tstart = vftr_get_runtime_usec();
   int retVal = PMPI_Iallreduce(sendbuf, recvbuf, count, datatype, op, comm, request);
   long long tend = vftr_get_runtime_usec();

   SELF_PROFILE_START_FUNCTION;
   long long t2start = tend;
   // Every process of group A performs the reduction within the group A
   // and stores the result on everyp process of group B and vice versa
   int size;
   PMPI_Comm_remote_size(comm, &size);
   // allocate memory for the temporary arrays
   // to register communication request
   int *tmpcount = (int*) malloc(sizeof(int)*size);
   MPI_Datatype *tmptype = (MPI_Datatype*) malloc(sizeof(MPI_Datatype)*size);
   int *peer_ranks = (int*) malloc(sizeof(int)*size);
   // messages to be send
   for (int i=0; i<size; i++) {
      tmpcount[i] = count;
      tmptype[i] = datatype;
      // translate the i-th rank in the remote group to the global rank
      peer_ranks[i] = vftr_remote2global_rank(comm, i);
   }
   // Register request with MPI_COMM_WORLD as communicator
   // to prevent additional (and thus faulty rank translation)
   vftr_register_collective_request(send, size, tmpcount, tmptype, peer_ranks,
                                    MPI_COMM_WORLD, *request, 0, NULL, tstart);
   // messages to be received
   for (int i=0; i<size; i++) {
      tmpcount[i] = count;
      tmptype[i] = datatype;
      // translate the i-th rank in the remote group to the global rank
      peer_ranks[i] = vftr_remote2global_rank(comm, i);
   }

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
   //
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
   stack_t *my_stack = vftrace.process.stacktree.stacks+my_threadstack->stackID;
   profile_t *my_profile = vftr_get_my_profile(my_stack, my_thread);
   long long t2end = vftr_get_runtime_usec();

   vftr_accumulate_mpiprofiling_overhead(&(my_profile->mpiProf), t2end-t2start);

   SELF_PROFILE_END_FUNCTION;
   return retVal;
}
