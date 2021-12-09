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

#include "neighbor_alltoall.h"

void vftr_MPI_Neighbor_alltoall_f082vftr(void *sendbuf, MPI_Fint *sendcount,
                                         MPI_Fint *f_sendtype, void *recvbuf,
                                         MPI_Fint *recvcount, MPI_Fint *f_recvtype,
                                         MPI_Fint *f_comm, MPI_Fint *f_error) {

   MPI_Datatype c_sendtype = PMPI_Type_f2c(*f_sendtype);
   MPI_Datatype c_recvtype = PMPI_Type_f2c(*f_recvtype);
   MPI_Comm c_comm = PMPI_Comm_f2c(*f_comm);

   int c_error;
   // determine the topology of the communicator
   int topology;
   PMPI_Topo_test(c_comm, &topology);
   switch(topology) {
      case MPI_GRAPH:
         c_error = vftr_MPI_Neighbor_alltoall_graph(sendbuf,
                                                    (int)(*sendcount),
                                                    c_sendtype,
                                                    recvbuf,
                                                    (int)(*recvcount),
                                                    c_recvtype,
                                                    c_comm);
         break;
      case MPI_CART:
         c_error = vftr_MPI_Neighbor_alltoall_cart(sendbuf,
                                                   (int)(*sendcount),
                                                   c_sendtype,
                                                   recvbuf,
                                                   (int)(*recvcount),
                                                   c_recvtype,
                                                   c_comm);
         break;
      case MPI_DIST_GRAPH:
         c_error = vftr_MPI_Neighbor_alltoall_dist_graph(sendbuf,
                                                         (int)(*sendcount),
                                                         c_sendtype,
                                                         recvbuf,
                                                         (int)(*recvcount),
                                                         c_recvtype,
                                                         c_comm);
         break;
      case MPI_UNDEFINED:
      default:
         c_error = PMPI_Neighbor_alltoall(sendbuf,
                                          (int)(*sendcount),
                                          c_sendtype,
                                          recvbuf,
                                          (int)(*recvcount),
                                          c_recvtype,
                                          c_comm);
   }

   *f_error = (MPI_Fint) (c_error);
}

#endif
