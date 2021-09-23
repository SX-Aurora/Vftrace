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

#include "vftr_mpi_buf_addr_const.h"
#include "reduce_scatter.h"

int vftr_MPI_Reduce_scatter_c2vftr(const void *sendbuf, void *recvbuf,
                                   const int *recvcounts, MPI_Datatype datatype,
                                   MPI_Op op, MPI_Comm comm) {
   // determine if inter or intra communicator
   int isintercom;
   PMPI_Comm_test_inter(comm, &isintercom);
   if (isintercom) {
      return vftr_MPI_Reduce_scatter_intercom(sendbuf, recvbuf,
                                              recvcounts, datatype,
                                              op, comm);
   } else {
      if (vftr_is_C_MPI_IN_PLACE(sendbuf)) {
        return vftr_MPI_Reduce_scatter_inplace(sendbuf, recvbuf,
                                               recvcounts, datatype,
                                               op, comm);
      } else {
        return vftr_MPI_Reduce_scatter(sendbuf, recvbuf,
                                       recvcounts, datatype,
                                       op, comm);
      }
   }
}

#endif
