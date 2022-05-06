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
#include <stdlib.h>

#include <mpi.h>

#include "ineighbor_alltoallv.h"

int vftr_MPI_Ineighbor_alltoallv_c2vftr(const void *sendbuf, const int *sendcounts,
                                        const int *sdispls, MPI_Datatype sendtype,
                                        void *recvbuf, const int *recvcounts,
                                        const int *rdispls, MPI_Datatype recvtype,
                                        MPI_Comm comm, MPI_Request *request) {
   // create a copy of recvcount and displs
   // It will be deallocated upon completion of the request
   int sizein;
   int sizeout;
   // determine the topology of the communicator
   int topology;
   PMPI_Topo_test(comm, &topology);
   switch(topology) {
      case MPI_GRAPH:;
         int rank;
         PMPI_Comm_rank(comm, &rank);
         PMPI_Graph_neighbors_count(comm, rank, &sizein);
         sizeout = sizein;
         break;
      case MPI_CART:
         PMPI_Cartdim_get(comm, &sizein);
         // Number of neighbors for cartesian communicators is always 2*ndims
         sizein *= 2;
         sizeout = sizein;
         break;
      case MPI_DIST_GRAPH:;
         int weighted;
         PMPI_Dist_graph_neighbors_count(comm, &sizein,
                                         &sizeout, &weighted);
         break;
   }

   int *tmp_sendcounts = (int*) malloc(sizeout*sizeof(int));
   for (int i=0; i<sizeout; i++) {
      tmp_sendcounts[i] = sendcounts[i];
   }
   int *tmp_sdispls = (int*) malloc(sizeout*sizeof(int));
   for (int i=0; i<sizeout; i++) {
      tmp_sdispls[i] = sdispls[i];
   }

   int *tmp_recvcounts = (int*) malloc(sizein*sizeof(int));
   for (int i=0; i<sizein; i++) {
      tmp_recvcounts[i] = recvcounts[i];
   }
   int *tmp_rdispls = (int*) malloc(sizein*sizeof(int));
   for (int i=0; i<sizein; i++) {
      tmp_rdispls[i] = rdispls[i];
   }

   switch(topology) {
      case MPI_GRAPH:
         return vftr_MPI_Ineighbor_alltoallv_graph(sendbuf, tmp_sendcounts,
                                                   tmp_sdispls,sendtype,
                                                   recvbuf, tmp_recvcounts,
                                                   tmp_rdispls, recvtype,
                                                   comm, request);
         break;
      case MPI_CART:
         return vftr_MPI_Ineighbor_alltoallv_cart(sendbuf, tmp_sendcounts,
                                                  tmp_sdispls, sendtype,
                                                  recvbuf, tmp_recvcounts,
                                                  tmp_rdispls, recvtype,
                                                  comm, request);
         break;
      case MPI_DIST_GRAPH:
         return vftr_MPI_Ineighbor_alltoallv_dist_graph(sendbuf, tmp_sendcounts,
                                                        tmp_sdispls, sendtype,
                                                        recvbuf, tmp_recvcounts,
                                                        tmp_rdispls, recvtype,
                                                        comm, request);
         break;
      case MPI_UNDEFINED:
      default:
         // should never get here.
         // But if so, free the arrays
         free(tmp_sendcounts);
         free(tmp_sdispls);
         free(tmp_recvcounts);
         free(tmp_rdispls);
         return PMPI_Ineighbor_alltoallv(sendbuf, sendcounts,
                                         sdispls, sendtype,
                                         recvbuf, recvcounts,
                                         rdispls, recvtype,
                                         comm, request);
   }
}
