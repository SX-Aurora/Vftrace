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

#include "vftr_timer.h"
#include "vftr_sync_messages.h"

int vftr_MPI_Alltoallv(const void *sendbuf, const int *sendcounts,
                       const int *sdispls, MPI_Datatype sendtype,
                       void *recvbuf, const int *recvcounts,
                       const int *rdispls, MPI_Datatype recvtype,
                       MPI_Comm comm) {
   long long tstart = vftr_get_runtime_usec();
   int retVal = PMPI_Alltoallv(sendbuf, sendcounts, sdispls, sendtype,
                               recvbuf, recvcounts, rdispls, recvtype, comm);
   long long tend = vftr_get_runtime_usec();

   long long t2start = tend;
   int size;
   PMPI_Comm_size(comm, &size);
   for (int i=0; i<size; i++) {
      vftr_store_sync_message_info(send, sendcounts[i], sendtype, i, 0,
                                   comm, tstart, tend);
      vftr_store_sync_message_info(recv, recvcounts[i], recvtype, i, 0,
                                   comm, tstart, tend);
   }
   long long t2end = vftr_get_runtime_usec();

   vftr_mpi_overhead_usec += t2end - t2start;

   return retVal;
}

int vftr_MPI_Alltoallv_inplace(const void *sendbuf, const int *sendcounts,
                               const int *sdispls, MPI_Datatype sendtype,
                               void *recvbuf, const int *recvcounts,
                               const int *rdispls, MPI_Datatype recvtype,
                               MPI_Comm comm) {
   long long tstart = vftr_get_runtime_usec();
   int retVal = PMPI_Alltoallv(sendbuf, sendcounts, sdispls, sendtype,
                               recvbuf, recvcounts, rdispls, recvtype, comm);
   long long tend = vftr_get_runtime_usec();

   long long t2start = tend;
   int size;
   PMPI_Comm_size(comm, &size);
   sendcounts = recvcounts;
   sendtype = recvtype;
   // For the in-place option no self communication is executed
   int rank;
   PMPI_Comm_rank(comm, &rank);
   for (int i=0; i<rank; i++) {
      vftr_store_sync_message_info(send, sendcounts[i], sendtype, i, 0,
                                   comm, tstart, tend);
      vftr_store_sync_message_info(recv, recvcounts[i], recvtype, i, 0,
                                   comm, tstart, tend);
   }
   for (int i=rank+1; i<size; i++) {
      vftr_store_sync_message_info(send, sendcounts[i], sendtype, i, 0,
                                   comm, tstart, tend);
      vftr_store_sync_message_info(recv, recvcounts[i], recvtype, i, 0,
                                   comm, tstart, tend);
   }
   long long t2end = vftr_get_runtime_usec();

   vftr_mpi_overhead_usec += t2end - t2start;

   return retVal;
}

int vftr_MPI_Alltoallv_intercom(const void *sendbuf, const int *sendcounts,
                                const int *sdispls, MPI_Datatype sendtype,
                                void *recvbuf, const int *recvcounts,
                                const int *rdispls, MPI_Datatype recvtype,
                                MPI_Comm comm) {
   long long tstart = vftr_get_runtime_usec();
   int retVal = PMPI_Alltoallv(sendbuf, sendcounts, sdispls, sendtype,
                               recvbuf, recvcounts, rdispls, recvtype, comm);
   long long tend = vftr_get_runtime_usec();

   long long t2start = tend;
   // Every process of group A sends sendcounts[i] sendtypes to
   // and receives recvcounts[i] recvtypes from
   // the i-th process in group B and vice versa.
   int size;
   PMPI_Comm_remote_size(comm, &size);
   for (int i=0; i<size; i++) {
      // translate the i-th rank in the remote group to the global rank
      int global_peer_rank = vftr_remote2global_rank(comm, i);
      // Store message info with MPI_COMM_WORLD as communicator
      // to prevent additional (and thus faulty rank translation)
      vftr_store_sync_message_info(send, sendcounts[i], sendtype,
                                   global_peer_rank, -1, MPI_COMM_WORLD,
                                   tstart, tend);
      vftr_store_sync_message_info(recv, recvcounts[i], recvtype,
                                   global_peer_rank, -1, MPI_COMM_WORLD,
                                   tstart, tend);
   }
   long long t2end = vftr_get_runtime_usec();

   vftr_mpi_overhead_usec += t2end - t2start;

   return retVal;
}
