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

#include "vftr_mpi_utils.h"
#include "ialltoallw.h"

int vftr_MPI_Ialltoallw(const void *sendbuf, const int *sendcounts,
                   const int *sdispls, const MPI_Datatype *sendtypes,
                   void *recvbuf, const int *recvcounts, const int *rdispls,
                   const MPI_Datatype *recvtypes, MPI_Comm comm,
                   MPI_Request *request) {
   if (vftr_no_mpi_logging()) {
      return PMPI_Ialltoallw(sendbuf, sendcounts, sdispls, sendtypes,
                             recvbuf, recvcounts, rdispls, recvtypes, comm,
                             request);
   } else {
      int size;
      int isintercom;
      PMPI_Comm_test_inter(comm, &isintercom);
      if (isintercom) {
         PMPI_Comm_remote_size(comm, &size);
      } else {
         PMPI_Comm_size(comm, &size);
      }
  
      int *tmp_sendcounts = (int*) malloc(size*sizeof(int));
      for (int i=0; i<size; i++) {
         tmp_sendcounts[i] = sendcounts[i];
      }
      int *tmp_sdispls = (int*) malloc(size*sizeof(int));
      for (int i=0; i<size; i++) {
         tmp_sdispls[i] = sdispls[i];
      }
      MPI_Datatype *tmp_sendtypes = (MPI_Datatype*) malloc(size*sizeof(MPI_Datatype));
      for (int i=0; i<size; i++) {
         tmp_sendtypes[i] = sendtypes[i];
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

      return vftr_MPI_Ialltoallw(sendbuf, tmp_sendcounts, tmp_sdispls, tmp_sendtypes,
                                 recvbuf, tmp_recvcounts, tmp_rdispls, tmp_recvtypes,
                                 comm, request);
   }
}

#endif
