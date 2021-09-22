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

#include "vftr_buf_addr_const.h"
#include "iscatterv.h"

int vftr_MPI_Iscatterv_c2vftr(const void *sendbuf, const int *sendcounts,
                              const int *displs, MPI_Datatype sendtype,
                              void *recvbuf, int recvcount,
                              MPI_Datatype recvtype, int root,
                              MPI_Comm comm, MPI_Request *request) {
   int isroot;
   int size;
   int isintercom;
   PMPI_Comm_test_inter(comm, &isintercom);
   if (isintercom) {
      isroot = MPI_ROOT == root;
   } else {
      int myrank;
      PMPI_Comm_rank(comm, &myrank);
      isroot = myrank == root;
   }

   int *tmp_sendcounts = NULL;
   int *tmp_displs = NULL;
   if (isroot) {
      if (isintercom) {
         PMPI_Comm_remote_size(comm, &size);
      } else {
         PMPI_Comm_size(comm, &size);
      }
      tmp_sendcounts = (int*) malloc(size*sizeof(int));
      for (int i=0; i<size; i++) {
         tmp_sendcounts[i] = sendcounts[i];
      }
      tmp_displs = (int*) malloc(size*sizeof(int));
      for (int i=0; i<size; i++) {
         tmp_displs[i] = displs[i];
      }
   }

   if (isintercom) {
      return vftr_MPI_Iscatterv_intercom(sendbuf, tmp_sendcounts, tmp_displs,
                                         sendtype, recvbuf, recvcount,
                                         recvtype, root, comm, request);
   } else {
      if (vftr_is_C_MPI_IN_PLACE(recvbuf)) {
         return vftr_MPI_Iscatterv_inplace(sendbuf, tmp_sendcounts, tmp_displs,
                                           sendtype, recvbuf, recvcount,
                                           recvtype, root, comm, request);
      } else {
         return vftr_MPI_Iscatterv(sendbuf, tmp_sendcounts, tmp_displs,
                                   sendtype, recvbuf, recvcount,
                                   recvtype, root, comm, request);
      }
   }
}

#endif
