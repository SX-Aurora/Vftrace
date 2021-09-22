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
#include <stdlib.h>

#include <mpi.h>

#include "vftr_mpi_utils.h"
#include "iscatterv_c2vftr.h"

int MPI_Iscatterv(const void *sendbuf, const int *sendcounts,
                  const int *displs, MPI_Datatype sendtype,
                  void *recvbuf, int recvcount,
                  MPI_Datatype recvtype, int root,
                  MPI_Comm comm, MPI_Request *request) {
   if (vftr_no_mpi_logging()) {
      return PMPI_Iscatterv(sendbuf, sendcounts, displs, sendtype,
                            recvbuf, recvcount, recvtype, root, comm,
                            request);
   } else {
      return vftr_MPI_Iscatterv_c2vftr(sendbuf, sendcounts, displs,
                                       sendtype, recvbuf, recvcount,
                                       recvtype, root, comm, request);
   }
}

#endif
