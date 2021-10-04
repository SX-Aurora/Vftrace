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

#include "mpi_buf_addr_const.h"
#include "ialltoallw.h"

int vftr_MPI_Ialltoallw_c2vftr(const void *sendbuf, const int *sendcounts,
                               const int *sdispls, const MPI_Datatype *sendtypes,
                               void *recvbuf, const int *recvcounts,
                               const int *rdispls, const MPI_Datatype *recvtypes,
                               MPI_Comm comm, MPI_Request *request) {
   int size;
   int isintercom;
   PMPI_Comm_test_inter(comm, &isintercom);
   if (isintercom) {
      PMPI_Comm_remote_size(comm, &size);
   } else {
      PMPI_Comm_size(comm, &size);
   }

   int *tmp_sendcounts = NULL;
   int *tmp_sdispls = NULL;
   MPI_Datatype *tmp_sendtypes = NULL;
   // sendcounts, senddisplacements, and sendtypes are ignored
   // if sendbuffer is MPI_IN_PLACE
   if (!vftr_is_C_MPI_IN_PLACE(sendbuf)) {
      tmp_sendcounts = (int*) malloc(size*sizeof(int));
      for (int i=0; i<size; i++) {
         tmp_sendcounts[i] = sendcounts[i];
      }
      tmp_sdispls = (int*) malloc(size*sizeof(int));
      for (int i=0; i<size; i++) {
         tmp_sdispls[i] = sdispls[i];
      }
      tmp_sendtypes = (MPI_Datatype*) malloc(size*sizeof(MPI_Datatype));
      for (int i=0; i<size; i++) {
         tmp_sendtypes[i] = sendtypes[i];
      }
   }

   int *tmp_recvcounts = (int*) malloc(size*sizeof(int));
   for (int i=0; i<size; i++) {
      tmp_recvcounts[i] = recvcounts[i];
   }
   int *tmp_rdispls = (int*) malloc(size*sizeof(int));
   for (int i=0; i<size; i++) {
      tmp_rdispls[i] = rdispls[i];
   }
   MPI_Datatype *tmp_recvtypes = (MPI_Datatype*) malloc(size*sizeof(MPI_Datatype));
   for (int i=0; i<size; i++) {
      tmp_recvtypes[i] = recvtypes[i];
   }

   if (isintercom) {
      return vftr_MPI_Ialltoallw_intercom(sendbuf, tmp_sendcounts,
                                          tmp_sdispls, tmp_sendtypes,
                                          recvbuf, tmp_recvcounts,
                                          tmp_rdispls, tmp_recvtypes,
                                          comm, request);
   } else {
      if (vftr_is_C_MPI_IN_PLACE(sendbuf)) {
         return vftr_MPI_Ialltoallw_inplace(sendbuf, tmp_sendcounts,
                                            tmp_sdispls, tmp_sendtypes,
                                            recvbuf, tmp_recvcounts,
                                            tmp_rdispls, tmp_recvtypes,
                                            comm, request);
      } else {
         return vftr_MPI_Ialltoallw(sendbuf, tmp_sendcounts,
                                    tmp_sdispls, tmp_sendtypes,
                                    recvbuf, tmp_recvcounts,
                                    tmp_rdispls, tmp_recvtypes,
                                    comm, request);
      }
   }
}

#endif
