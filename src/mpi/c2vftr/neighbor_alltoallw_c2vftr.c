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
#include "neighbor_alltoallw.h"

int vftr_MPI_Neighbor_alltoallw_c2vftr(const void *sendbuf, const int *sendcounts,
                                       const MPI_Aint *sdispls, const MPI_Datatype *sendtypes,
                                       void *recvbuf, const int *recvcounts,
                                       const MPI_Aint *rdispls, const MPI_Datatype *recvtypes,
                                       MPI_Comm comm) {
   // determine the topology of the communicator
   int topology;
   PMPI_Topo_test(comm, &topology);
   switch(topology) {
      case MPI_GRAPH:
         return vftr_MPI_Neighbor_alltoallw_graph(sendbuf, sendcounts, sdispls, sendtypes,
                                                  recvbuf, recvcounts, rdispls, recvtypes,
                                                  comm);
         break;
      case MPI_CART:
         return vftr_MPI_Neighbor_alltoallw_cart(sendbuf, sendcounts, sdispls, sendtypes,
                                                 recvbuf, recvcounts, rdispls, recvtypes,
                                                 comm);
         break;
      case MPI_DIST_GRAPH:
         return vftr_MPI_Neighbor_alltoallw_dist_graph(sendbuf, sendcounts, sdispls, sendtypes,
                                                       recvbuf, recvcounts, rdispls, recvtypes,
                                                       comm);
         break;
      case MPI_UNDEFINED:
      default:
         return PMPI_Neighbor_alltoallw(sendbuf, sendcounts, sdispls, sendtypes,
                                        recvbuf, recvcounts, rdispls, recvtypes,
                                        comm);
   }
}

#endif
