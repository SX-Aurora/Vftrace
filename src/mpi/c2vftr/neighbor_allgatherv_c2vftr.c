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

#ifdef _MPI
#include <mpi.h>

#include "mpi_buf_addr_const.h"
#include "neighbor_allgatherv.h"

int vftr_MPI_Neighbor_allgather_c2vftr(const void *sendbuf, int sendcount,
                                       MPI_Datatype sendtype, void *recvbuf,
                                       const int *recvcounts, const int *displs,
                                       MPI_Datatype recvtype, MPI_Comm comm) {
   // determine the topology of the communicator
   int topology;
   PMPI_Topo_test(comm, &topology);
   switch(topology) {
      case MPI_GRAPH:
         return vftr_MPI_Neighbor_allgatherv_graph(sendbuf, sendcount, sendtype,
                                                   recvbuf, recvcounts, displs, recvtype,
                                                   comm);
         break;
      case MPI_CART:
         return vftr_MPI_Neighbor_allgatherv_cart(sendbuf, sendcount, sendtype,
                                                  recvbuf, recvcounts, displs, recvtype,
                                                  comm);
         break;
      case MPI_DIST_GRAPH:
         return vftr_MPI_Neighbor_allgatherv_dist_graph(sendbuf, sendcount, sendtype,
                                                        recvbuf, recvcounts, displs, recvtype,
                                                        comm);
         break;
      case MPI_UNDEFINED:
      default:
         return PMPI_Neighbor_allgatherv(sendbuf, sendcount, sendtype,
                                        recvbuf, recvcounts, displs, recvtype,
                                        comm);
   }
}

#endif
