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

int vftr_MPI_Igatherv(const void *sendbuf, int sendcount,
                      MPI_Datatype sendtype, void *recvbuf,
                      const int *recvcounts, const int *displs,
                      MPI_Datatype recvtype, int root,
                      MPI_Comm comm, MPI_Request *request) {
   long long tstart = vftr_get_runtime_usec();
   int retVal = PMPI_Igatherv(sendbuf, sendcount, sendtype, recvbuf,
                              recvcounts, displs, recvtype, root, comm, request);

   SELF_PROFILE_START_FUNCTION;
   long long t2start = vftr_get_runtime_usec();
   // in intracommunicators the expected behaviour is to
   // bcast from root to all other processes in the communicator
   int rank;
   PMPI_Comm_rank(comm, &rank);
   if (rank == root) {
      int size;
      PMPI_Comm_size(comm, &size);
      // Store tmp-pointers for delayed deallocation
      int n_tmp_ptr = 2;
      void **tmp_ptrs = (void**) malloc(n_tmp_ptr*sizeof(void*));
      tmp_ptrs[0] = (void*) recvcounts;
      tmp_ptrs[1] = (void*) displs;
      // self communication of root process
      vftr_register_collective_request(send, 1, &sendcount, &sendtype, &root,
                                       comm, *request, n_tmp_ptr,
                                       tmp_ptrs, tstart);

      // allocate memory for the temporary arrays
      // to register communication request
      int *tmprecvcounts = (int*) malloc(sizeof(int)*size);
      MPI_Datatype *tmprecvtype = (MPI_Datatype*) malloc(sizeof(MPI_Datatype)*size);
      int *tmppeer_ranks= (int*) malloc(sizeof(int)*size);
      // messeges to be received
      for (int i=0; i<size; i++) {
         tmprecvcounts[i] = recvcounts[i];
         tmprecvtype[i] = recvtype;
         tmppeer_ranks[i] = i;
      }
      vftr_register_collective_request(recv, size, tmprecvcounts, tmprecvtype,
                                       tmppeer_ranks, comm, *request,
                                       0, NULL, tstart);
      // cleanup temporary arrays
      free(tmprecvcounts);
      tmprecvcounts = NULL;
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
   long long t2end = vftr_get_runtime_usec();

   vftr_accumulate_mpiprofiling_overhead(&(my_profile->mpiProf), t2end-t2start);

   SELF_PROFILE_END_FUNCTION;
   return retVal;
}

int vftr_MPI_Igatherv_inplace(const void *sendbuf, int sendcount,
                              MPI_Datatype sendtype, void *recvbuf,
                              const int *recvcounts, const int *displs,
                              MPI_Datatype recvtype, int root,
                              MPI_Comm comm, MPI_Request *request) {
   long long tstart = vftr_get_runtime_usec();
   int retVal = PMPI_Igatherv(sendbuf, sendcount, sendtype, recvbuf,
                              recvcounts, displs, recvtype, root, comm, request);

   SELF_PROFILE_START_FUNCTION;
   long long t2start = vftr_get_runtime_usec();
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
         int *tmprecvcounts = (int*) malloc(sizeof(int)*(size-1));
         MPI_Datatype *tmprecvtype = (MPI_Datatype*) malloc(sizeof(MPI_Datatype)*(size-1));
         int *tmppeer_ranks= (int*) malloc(sizeof(int)*(size-1));
         // messeges to be received
         int idx = 0;
         for (int i=0; i<rank; i++) {
            tmprecvcounts[idx] = recvcounts[i];
            tmprecvtype[idx] = recvtype;
            tmppeer_ranks[idx] = i;
            idx++;
         }
         for (int i=rank+1; i<size; i++) {
            tmprecvcounts[idx] = recvcounts[i];
            tmprecvtype[idx] = recvtype;
            tmppeer_ranks[idx] = i;
            idx++;
         }
         // Store tmp-pointers for delayed deallocation
         int n_tmp_ptr = 2;
         void **tmp_ptrs = (void**) malloc(n_tmp_ptr*sizeof(void*));
         tmp_ptrs[0] = (void*) recvcounts;
         tmp_ptrs[1] = (void*) displs;
         vftr_register_collective_request(recv, size-1, tmprecvcounts, tmprecvtype,
                                          tmppeer_ranks, comm, *request,
                                          n_tmp_ptr, tmp_ptrs, tstart);
         // cleanup temporary arrays
         free(tmprecvcounts);
         tmprecvcounts = NULL;
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
   long long t2end = vftr_get_runtime_usec();

   vftr_accumulate_mpiprofiling_overhead(&(my_profile->mpiProf), t2end-t2start);

   SELF_PROFILE_END_FUNCTION;
   return retVal;
}

int vftr_MPI_Igatherv_intercom(const void *sendbuf, int sendcount,
                               MPI_Datatype sendtype, void *recvbuf,
                               const int *recvcounts, const int *displs,
                               MPI_Datatype recvtype, int root,
                               MPI_Comm comm, MPI_Request *request) {
   long long tstart = vftr_get_runtime_usec();
   int retVal = PMPI_Igatherv(sendbuf, sendcount, sendtype, recvbuf,
                              recvcounts, displs, recvtype, root, comm, request);

   SELF_PROFILE_START_FUNCTION;
   long long t2start = vftr_get_runtime_usec();
   // in intercommunicators the behaviour is more complicated
   // There are two groups A and B
   // In group A the root process is located.
   if (root == MPI_ROOT) {
      // The root process get the special process wildcard MPI_ROOT
      // get the size of group B
      int size;
      PMPI_Comm_remote_size(comm, &size);
      int *tmprecvcounts = (int*) malloc(sizeof(int)*size);
      MPI_Datatype *tmprecvtype = (MPI_Datatype*) malloc(sizeof(MPI_Datatype)*size);
      int *tmppeer_ranks= (int*) malloc(sizeof(int)*size);
      // messages to be received
      for (int i=0; i<size; i++) {
         tmprecvcounts[i] = recvcounts[i];
         tmprecvtype[i] = recvtype;
         // translate the i-th rank in group B to the global rank
         tmppeer_ranks[i] = vftr_remote2global_rank(comm, i);
      }
      // Store tmp-pointers for delayed deallocation
      int n_tmp_ptr = 2;
      void **tmp_ptrs = (void**) malloc(n_tmp_ptr*sizeof(void*));
      tmp_ptrs[0] = (void*) recvcounts;
      tmp_ptrs[1] = (void*) displs;
      // Register request with MPI_COMM_WORLD as communicator
      // to prevent additional (and thus faulty rank translation)
      vftr_register_collective_request(recv, size, tmprecvcounts, tmprecvtype,
                                       tmppeer_ranks, MPI_COMM_WORLD,
                                       *request, n_tmp_ptr, tmp_ptrs, tstart);
      // cleanup temporary arrays
      free(tmprecvcounts);
      tmprecvcounts = NULL;
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
   long long t2end = vftr_get_runtime_usec();

   vftr_accumulate_mpiprofiling_overhead(&(my_profile->mpiProf), t2end-t2start);

   SELF_PROFILE_END_FUNCTION;
   return retVal;
}
