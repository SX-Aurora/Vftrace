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

int vftr_MPI_Ialltoallw(const void *sendbuf, const int *sendcounts,
                        const int *sdispls, const MPI_Datatype *sendtypes,
                        void *recvbuf, const int *recvcounts,
                        const int *rdispls, const MPI_Datatype *recvtypes,
                        MPI_Comm comm, MPI_Request *request) {
   long long tstart = vftr_get_runtime_nsec();
   int retVal = PMPI_Ialltoallw(sendbuf, sendcounts, sdispls, sendtypes,
                                recvbuf, recvcounts, rdispls, recvtypes,
                                comm, request);

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
      tmpcount[i] = sendcounts[i];
      tmptype[i] = sendtypes[i];
      peer_ranks[i] = i;
   }
   int n_tmp_ptr = 3;
   void **tmp_ptrs = (void**) malloc(n_tmp_ptr*sizeof(void*));
   tmp_ptrs[0] = (void*) sendcounts;
   tmp_ptrs[1] = (void*) sdispls;
   tmp_ptrs[2] = (void*) sendtypes;
   vftr_register_collective_request(send, size, tmpcount, tmptype, peer_ranks,
                                    comm, *request, n_tmp_ptr, tmp_ptrs, tstart);
   // messages to be received
   for (int i=0; i<size; i++) {
      tmpcount[i] = recvcounts[i];
      tmptype[i] = recvtypes[i];
      peer_ranks[i] = i;
   }
   tmp_ptrs = (void**) malloc(n_tmp_ptr*sizeof(void*));
   tmp_ptrs[0] = (void*) recvcounts;
   tmp_ptrs[1] = (void*) rdispls;
   tmp_ptrs[2] = (void*) recvtypes;
   vftr_register_collective_request(recv, size, tmpcount, tmptype, peer_ranks,
                                    comm, *request, n_tmp_ptr, tmp_ptrs, tstart);
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
   long long t2end = vftr_get_runtime_nsec();

   vftr_accumulate_mpiprofiling_overhead(&(my_profile->mpiProf), t2end-t2start);

   SELF_PROFILE_END_FUNCTION;
   return retVal;
}

int vftr_MPI_Ialltoallw_inplace(const void *sendbuf, const int *sendcounts,
                                const int *sdispls, const MPI_Datatype *sendtypes,
                                void *recvbuf, const int *recvcounts,
                                const int *rdispls, const MPI_Datatype *recvtypes,
                                MPI_Comm comm, MPI_Request *request) {
   long long tstart = vftr_get_runtime_nsec();
   int retVal = PMPI_Ialltoallw(sendbuf, sendcounts, sdispls, sendtypes,
                                recvbuf, recvcounts, rdispls, recvtypes,
                                comm, request);

   SELF_PROFILE_START_FUNCTION;
   long long t2start = vftr_get_runtime_nsec();
   int size;
   PMPI_Comm_size(comm, &size);
   if (size > 1) {
      int rank;
      PMPI_Comm_rank(comm, &rank);
      // For the in-place option no self communication is executed

      // allocate memory for the temporary arrays
      // to register communication request
      int *tmpcount = (int*) malloc(sizeof(int)*(size-1));
      MPI_Datatype *tmptype = (MPI_Datatype*)
                              malloc(sizeof(MPI_Datatype)*(size-1));
      int *peer_ranks = (int*) malloc(sizeof(int)*(size-1));
      // messages to be send
      int idx = 0;
      for (int i=0; i<rank; i++) {
         tmpcount[idx] = recvcounts[i];
         tmptype[idx] = recvtypes[i];
         peer_ranks[idx] = i;
         idx++;
      }
      for (int i=rank+1; i<size; i++) {
         tmpcount[idx] = recvcounts[i];
         tmptype[idx] = recvtypes[i];
         peer_ranks[idx] = i;
         idx++;
      }
      // Store tmp-pointers for delayed deallocation
      // For sendbuff = MPI_IN_PLACE sendcount senddisplacement and sendtype are ignored.
      int n_tmp_ptr = 0;
      void **tmp_ptrs = NULL;
      vftr_register_collective_request(send, size-1, tmpcount, tmptype, peer_ranks,
                                       comm, *request, n_tmp_ptr, tmp_ptrs, tstart);
      tmp_ptrs = (void**) malloc(n_tmp_ptr*sizeof(void*));
      tmp_ptrs[0] = (void*) recvcounts;
      tmp_ptrs[1] = (void*) rdispls;
      tmp_ptrs[2] = (void*) recvtypes;
      vftr_register_collective_request(recv, size-1, tmpcount, tmptype, peer_ranks,
                                       comm, *request, n_tmp_ptr, tmp_ptrs, tstart);
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
   long long t2end = vftr_get_runtime_nsec();

   vftr_accumulate_mpiprofiling_overhead(&(my_profile->mpiProf), t2end-t2start);

   SELF_PROFILE_END_FUNCTION;
   return retVal;
}

int vftr_MPI_Ialltoallw_intercom(const void *sendbuf, const int *sendcounts,
                                 const int *sdispls, const MPI_Datatype *sendtypes,
                                 void *recvbuf, const int *recvcounts,
                                 const int *rdispls, const MPI_Datatype *recvtypes,
                                 MPI_Comm comm, MPI_Request *request) {
   long long tstart = vftr_get_runtime_nsec();
   int retVal = PMPI_Ialltoallw(sendbuf, sendcounts, sdispls, sendtypes,
                                recvbuf, recvcounts, rdispls, recvtypes,
                                comm, request);

   SELF_PROFILE_START_FUNCTION;
   long long t2start = vftr_get_runtime_nsec();
   // Every process of group A sends sendcounts[i] sendtypes[i] to
   // and receives recvcounts[i] recvtypes[i] from
   // the i-th process in group B and vice versa.
   int size;
   PMPI_Comm_remote_size(comm, &size);
   // allocate memory for the temporary arrays
   // to register communication request
   int *tmpcount = (int*) malloc(sizeof(int)*size);
   MPI_Datatype *tmptype = (MPI_Datatype*) malloc(sizeof(MPI_Datatype)*size);
   int *peer_ranks = (int*) malloc(sizeof(int)*size);
   // messages to be send
   for (int i=0; i<size; i++) {
      tmpcount[i] = sendcounts[i];
      tmptype[i] = sendtypes[i];
      // translate the i-th rank in the remote group to the global rank
      peer_ranks[i] = vftr_remote2global_rank(comm, i);
   }
   // Store tmp-pointers for delayed deallocation
   int n_tmp_ptr = 3;
   void **tmp_ptrs = (void**) malloc(n_tmp_ptr*sizeof(void*));
   tmp_ptrs[0] = (void*) sendcounts;
   tmp_ptrs[1] = (void*) sdispls;
   tmp_ptrs[2] = (void*) sendtypes;
   // Register request with MPI_COMM_WORLD as communicator
   // to prevent additional (and thus faulty rank translation)
   vftr_register_collective_request(send, size, tmpcount, tmptype, peer_ranks,
                                    MPI_COMM_WORLD, *request,
                                    n_tmp_ptr, tmp_ptrs, tstart);
   // messages to be received
   for (int i=0; i<size; i++) {
      tmpcount[i] = recvcounts[i];
      tmptype[i] = recvtypes[i];
      // translate the i-th rank in the remote group to the global rank
      peer_ranks[i] = vftr_remote2global_rank(comm, i);
   }
   // Store tmp-pointers for delayed deallocation
   tmp_ptrs = (void**) malloc(n_tmp_ptr*sizeof(void*));
   tmp_ptrs[0] = (void*) recvcounts;
   tmp_ptrs[1] = (void*) rdispls;
   tmp_ptrs[2] = (void*) recvtypes;
   // Register request with MPI_COMM_WORLD as communicator
   // to prevent additional (and thus faulty rank translation)
   vftr_register_collective_request(recv, size, tmpcount, tmptype, peer_ranks,
                                    MPI_COMM_WORLD, *request,
                                    n_tmp_ptr, tmp_ptrs, tstart);
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
   long long t2end = vftr_get_runtime_nsec();

   vftr_accumulate_mpiprofiling_overhead(&(my_profile->mpiProf), t2end-t2start);

   SELF_PROFILE_END_FUNCTION;
   return retVal;
}
