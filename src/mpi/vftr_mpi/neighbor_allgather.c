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
#include "vftr_timer.h"
#include "sync_messages.h"
#include "cart_comms.h"

int vftr_MPI_Neighbor_allgather_graph(const void *sendbuf, int sendcount,
                                      MPI_Datatype sendtype, void *recvbuf,
                                      int recvcount, MPI_Datatype recvtype,
                                      MPI_Comm comm) {
   long long tstart = vftr_get_runtime_usec();
   int retVal = PMPI_Neighbor_allgather(sendbuf, sendcount, sendtype, recvbuf,
                                        recvcount, recvtype, comm);
   long long tend = vftr_get_runtime_usec();

   long long t2start = vftr_get_runtime_usec();
   int rank;
   PMPI_Comm_rank(comm, &rank);
   int nneighbors;
   MPI_Graph_neighbors_count(comm, rank, &nneighbors);
   int *neighbors = (int*) malloc(nneighbors*sizeof(int));
   MPI_Graph_neighbors(comm, rank, nneighbors, neighbors);
   for (int ineighbor=0; ineighbor<nneighbors; ineighbor++) {
      vftr_store_sync_message_info(send, sendcount, sendtype,
                                   neighbors[ineighbor], -1,
                                   comm, tstart, tend);
      vftr_store_sync_message_info(recv, sendcount, sendtype,
                                   neighbors[ineighbor], -1,
                                   comm, tstart, tend);
   }
   free(neighbors);
   neighbors = NULL;
   long long t2end = vftr_get_runtime_usec();

   vftr_mpi_overhead_usec += t2end - t2start;

   return retVal;
}

int vftr_MPI_Neighbor_allgather_cart(const void *sendbuf, int sendcount,
                                     MPI_Datatype sendtype, void *recvbuf,
                                     int recvcount, MPI_Datatype recvtype,
                                     MPI_Comm comm) {
   long long tstart = vftr_get_runtime_usec();
   int retVal = PMPI_Neighbor_allgather(sendbuf, sendcount, sendtype, recvbuf,
                                        recvcount, recvtype, comm);
   long long tend = vftr_get_runtime_usec();

   long long t2start = vftr_get_runtime_usec();
   int rank;
   PMPI_Comm_rank(comm, &rank);
   int nneighbors;
   int *neighbors;
   vftr_mpi_cart_neighbor_ranks(comm, &nneighbors, &neighbors);
   for (int ineighbor=0; ineighbor<nneighbors; ineighbor++) {
      // the neighborlist contains a -1 for non existent neighbors
      // due to non-periodic boundaries
      if (neighbors[ineighbor] >= 0) {
         vftr_store_sync_message_info(send, sendcount, sendtype,
                                      neighbors[ineighbor], -1,
                                      comm, tstart, tend);
         vftr_store_sync_message_info(recv, sendcount, sendtype,
                                      neighbors[ineighbor], -1,
                                      comm, tstart, tend);
      }
   }
   free(neighbors);
   neighbors = NULL;
   long long t2end = vftr_get_runtime_usec();

   vftr_mpi_overhead_usec += t2end - t2start;

   return retVal;
}

int vftr_MPI_Neighbor_allgather_dist_graph(const void *sendbuf, int sendcount,
                                           MPI_Datatype sendtype, void *recvbuf,
                                           int recvcount, MPI_Datatype recvtype,
                                           MPI_Comm comm) {
   long long tstart = vftr_get_runtime_usec();
   int retVal = PMPI_Neighbor_allgather(sendbuf, sendcount, sendtype, recvbuf,
                                        recvcount, recvtype, comm);
   long long tend = vftr_get_runtime_usec();

   long long t2start = vftr_get_runtime_usec();
//   // Every process of group A sends sendcount data to and
//   // receives recvcount data from every process in group B and
//   // vice versa
//   int size;
//   PMPI_Comm_remote_size(comm, &size);
//   for (int i=0; i<size; i++) {
//      // translate the i-th rank in the remote group to the global rank
//      int global_peer_rank = vftr_remote2global_rank(comm, i);
//      // Store message info with MPI_COMM_WORLD as communicator
//      // to prevent additional (and thus faulty rank translation)
//      vftr_store_sync_message_info(send, sendcount, sendtype, 
//                                   global_peer_rank, -1, MPI_COMM_WORLD,
//                                   tstart, tend);
//      vftr_store_sync_message_info(recv, recvcount, recvtype,
//                                   global_peer_rank, -1, MPI_COMM_WORLD,
//                                   tstart, tend);
//   }
   long long t2end = vftr_get_runtime_usec();

   vftr_mpi_overhead_usec += t2end - t2start;

   return retVal;
}
