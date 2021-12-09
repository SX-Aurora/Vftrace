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

#include <stdlib.h>

#include "neighbor_allgatherv.h"

void vftr_MPI_Neighbor_allgatherv_f082vftr(void *sendbuf, MPI_Fint *sendcount,
                                           MPI_Fint *f_sendtype, void *recvbuf,
                                           MPI_Fint *f_recvcounts, MPI_Fint *f_displs, 
                                           MPI_Fint *f_recvtype, MPI_Fint *f_comm,
                                           MPI_Fint *f_error) {

   MPI_Comm c_comm = PMPI_Comm_f2c(*f_comm);

   // translate recvcount and displs to c-int
   int sizein;
   // determine the topology of the communicator
   int topology;
   PMPI_Topo_test(c_comm, &topology);
   switch(topology) {
      case MPI_GRAPH:;
         int rank;
         PMPI_Comm_rank(c_comm, &rank);
         MPI_Graph_neighbors_count(c_comm, rank, &sizein);
         break;
      case MPI_CART:
         MPI_Cartdim_get(c_comm, &sizein);
         // Number of neighbors for cartesian communicators is always 2*ndims
         sizein *= 2;
         break;
      case MPI_DIST_GRAPH:;
         int sizeout;
         int weighted;
         PMPI_Dist_graph_neighbors_count(c_comm, &sizein,
                                         &sizeout, &weighted);
         break;
   }
 
   int *c_recvcounts = (int*) malloc(sizein*sizeof(int));
   for (int i=0; i<sizein; i++) {
      c_recvcounts[i] = (int) f_recvcounts[i];
   }
   int *c_displs = (int*) malloc(sizein*sizeof(int));
   for (int i=0; i<sizein; i++) {
      c_displs[i] = (int) f_displs[i];
   }

   MPI_Datatype c_sendtype = PMPI_Type_f2c(*f_sendtype);
   MPI_Datatype c_recvtype = PMPI_Type_f2c(*f_recvtype);

   int c_error;
   switch(topology) {
      case MPI_GRAPH:
         c_error = vftr_MPI_Neighbor_allgatherv_graph(sendbuf,
                                                      (int)(*sendcount),
                                                      c_sendtype,
                                                      recvbuf,
                                                      c_recvcounts,
                                                      c_displs,
                                                      c_recvtype,
                                                      c_comm);
         break;
      case MPI_CART:
         c_error = vftr_MPI_Neighbor_allgatherv_cart(sendbuf,
                                                     (int)(*sendcount),
                                                     c_sendtype,
                                                     recvbuf,
                                                     c_recvcounts,
                                                     c_displs,
                                                     c_recvtype,
                                                     c_comm);
         break;
      case MPI_DIST_GRAPH:
         c_error = vftr_MPI_Neighbor_allgatherv_dist_graph(sendbuf,
                                                           (int)(*sendcount),
                                                           c_sendtype,
                                                           recvbuf,
                                                           c_recvcounts,
                                                           c_displs,
                                                           c_recvtype,
                                                           c_comm);
         break;
      case MPI_UNDEFINED:
      default:
         // should never get here.
         c_error = PMPI_Neighbor_allgatherv(sendbuf,
                                            (int)(*sendcount),
                                            c_sendtype,
                                            recvbuf,
                                            c_recvcounts,
                                            c_displs,
                                            c_recvtype,
                                            c_comm);
   }

   free(c_recvcounts);
   free(c_displs);

   *f_error = (MPI_Fint) (c_error);
}

#endif
