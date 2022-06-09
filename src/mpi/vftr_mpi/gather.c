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

#include "rank_translate.h"
#include "thread_types.h"
#include "threads.h"
#include "threadstack_types.h"
#include "threadstacks.h"
#include "stack_types.h"
#include "stacks.h"
#include "profiling_types.h"
#include "profiling.h"
#include "overheadprofiling.h"
#include "timer.h"
#include "sync_messages.h"

int vftr_MPI_Gather(const void *sendbuf, int sendcount,
                    MPI_Datatype sendtype, void *recvbuf,
                    int recvcount, MPI_Datatype recvtype,
                    int root, MPI_Comm comm) {
   long long tstart = vftr_get_runtime_usec();
   int retVal = PMPI_Gather(sendbuf, sendcount, sendtype, recvbuf, recvcount,
                            recvtype, root, comm);
   long long tend = vftr_get_runtime_usec();

   long long t2start = tend;
   // in intracommunicators the expected behaviour is to
   // bcast from root to all other processes in the communicator
   int rank;
   PMPI_Comm_rank(comm, &rank);
   if (rank == root) {
      int size;
      PMPI_Comm_size(comm, &size);
      // self communication of root process
      vftr_store_sync_message_info(send, sendcount, sendtype,
                                   root, -1, comm, tstart, tend);
      for (int i=0; i<size; i++) {
         vftr_store_sync_message_info(recv, recvcount, recvtype,
                                      i, -1, comm, tstart, tend);
      }
   } else {
      vftr_store_sync_message_info(send, sendcount, sendtype,
                                   root, -1, comm, tstart, tend);
   }

   thread_t *my_thread = vftr_get_my_thread(&(vftrace.process.threadtree));
   threadstack_t *my_threadstack = vftr_get_my_threadstack(my_thread);
   stack_t *my_stack = vftrace.process.stacktree.stacks+my_threadstack->stackID;
   profile_t *my_profile = vftr_get_my_profile(my_stack, my_thread);
   long long t2end = vftr_get_runtime_usec();

   vftr_accumulate_mpi_overheadprofiling(&(my_profile->overheadProf), t2end-t2start);

   return retVal;
}

int vftr_MPI_Gather_inplace(const void *sendbuf, int sendcount,
                            MPI_Datatype sendtype, void *recvbuf,
                            int recvcount, MPI_Datatype recvtype,
                            int root, MPI_Comm comm) {
   long long tstart = vftr_get_runtime_usec();
   int retVal = PMPI_Gather(sendbuf, sendcount, sendtype, recvbuf, recvcount,
                            recvtype, root, comm);
   long long tend = vftr_get_runtime_usec();

   long long t2start = tend;
   // in intracommunicators the expected behaviour is to
   // bcast from root to all other processes in the communicator
   int rank;
   PMPI_Comm_rank(comm, &rank);
   if (rank == root) {
      int size;
      PMPI_Comm_size(comm, &size);
      // For the in-place option no self communication is executed
      for (int i=0; i<rank; i++) {
         vftr_store_sync_message_info(recv, recvcount, recvtype,
                                      i, -1, comm, tstart, tend);
      }
      for (int i=rank+1; i<size; i++) {
         vftr_store_sync_message_info(recv, recvcount, recvtype,
                                      i, -1, comm, tstart, tend);
      }
   } else {
      vftr_store_sync_message_info(send, sendcount, sendtype,
                                   root, -1, comm, tstart, tend);
   }

   thread_t *my_thread = vftr_get_my_thread(&(vftrace.process.threadtree));
   threadstack_t *my_threadstack = vftr_get_my_threadstack(my_thread);
   stack_t *my_stack = vftrace.process.stacktree.stacks+my_threadstack->stackID;
   profile_t *my_profile = vftr_get_my_profile(my_stack, my_thread);
   long long t2end = vftr_get_runtime_usec();

   vftr_accumulate_mpi_overheadprofiling(&(my_profile->overheadProf), t2end-t2start);

   return retVal;
}

int vftr_MPI_Gather_intercom(const void *sendbuf, int sendcount,
                             MPI_Datatype sendtype, void *recvbuf,
                             int recvcount, MPI_Datatype recvtype,
                             int root, MPI_Comm comm) {
   long long tstart = vftr_get_runtime_usec();
   int retVal = PMPI_Gather(sendbuf, sendcount, sendtype, recvbuf, recvcount,
                            recvtype, root, comm);
   long long tend = vftr_get_runtime_usec();

   long long t2start = tend;
   // in intercommunicators the behaviour is more complicated
   // There are two groups A and B
   // In group A the root process is located.
   if (root == MPI_ROOT) {
      // The root process get the special process wildcard MPI_ROOT
      // get the size of group B
      int size;
      PMPI_Comm_remote_size(comm, &size);
      for (int i=0; i<size; i++) {
         // translate the i-th rank in group B to the global rank
         int global_peer_rank = vftr_remote2global_rank(comm, i);
         // store message info with MPI_COMM_WORLD as communicator
         // to prevent additional (and thus faulty rank translation)
         vftr_store_sync_message_info(recv, recvcount, recvtype,
                                      global_peer_rank, -1, MPI_COMM_WORLD,
                                      tstart, tend);
      }
   } else if (root == MPI_PROC_NULL) {
      // All other processes from group A pass wildcard MPI_PROC NULL
      // They do not participate in intercommunicator bcasts
      ;
   } else {
      // All other processes must be located in group B
      // root is the rank-id in group A Therefore no problems with
      // rank translation should arise
      vftr_store_sync_message_info(send, sendcount, sendtype,
                                   root, -1, comm, tstart, tend);
   }

   thread_t *my_thread = vftr_get_my_thread(&(vftrace.process.threadtree));
   threadstack_t *my_threadstack = vftr_get_my_threadstack(my_thread);
   stack_t *my_stack = vftrace.process.stacktree.stacks+my_threadstack->stackID;
   profile_t *my_profile = vftr_get_my_profile(my_stack, my_thread);
   long long t2end = vftr_get_runtime_usec();

   vftr_accumulate_mpi_overheadprofiling(&(my_profile->overheadProf), t2end-t2start);

   return retVal;
}
