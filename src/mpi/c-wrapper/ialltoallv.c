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
#include <stdio.h>

#include "vftr_mpi_utils.h"
#include "vftr_mpi_ialltoallv.h"

int MPI_Ialltoallv(const void *sendbuf, const int *sendcounts,
                   const int *sdispls, MPI_Datatype sendtype, void *recvbuf,
                   const int *recvcounts, const int *rdispls,
                   MPI_Datatype recvtype, MPI_Comm comm,
                   MPI_Request *request) {
printf("File: %s, Line: %d\n", __FILE__, __LINE__);
   if (vftr_no_mpi_logging()) {
printf("File: %s, Line: %d\n", __FILE__, __LINE__);
      return PMPI_Ialltoallv(sendbuf, sendcounts, sdispls, sendtype,
                             recvbuf, recvcounts, rdispls, recvtype, comm,
                             request);
printf("File: %s, Line: %d\n", __FILE__, __LINE__);
   } else {
printf("File: %s, Line: %d\n", __FILE__, __LINE__);
      int size;
      int isintercom;
printf("File: %s, Line: %d\n", __FILE__, __LINE__);
      PMPI_Comm_test_inter(comm, &isintercom);
      if (isintercom) {
         PMPI_Comm_remote_size(comm, &size);
      } else {
         PMPI_Comm_size(comm, &size);
      }
printf("File: %s, Line: %d\n", __FILE__, __LINE__);
   
      int *tmp_sendcounts = (int*) malloc(size*sizeof(int));
printf("File: %s, Line: %d, size %d, p=%p\n", __FILE__, __LINE__, size, sendcounts);
      for (int i=0; i<size; i++) {
         printf("%d: %d\n", i, sendcounts[i]);
         tmp_sendcounts[i] = sendcounts[i];
      }
printf("File: %s, Line: %d\n", __FILE__, __LINE__);
      int *tmp_sdispls = (int*) malloc(size*sizeof(int));
      for (int i=0; i<size; i++) {
         tmp_sdispls[i] = sdispls[i];
      }
printf("File: %s, Line: %d\n", __FILE__, __LINE__);
      int *tmp_recvcounts = (int*) malloc(size*sizeof(int));
      for (int i=0; i<size; i++) {
         tmp_recvcounts[i] = recvcounts[i];
      }
printf("File: %s, Line: %d\n", __FILE__, __LINE__);
      int *tmp_rdispls = (int*) malloc(size*sizeof(int));
      for (int i=0; i<size; i++) {
         tmp_rdispls[i] = rdispls[i];
      }
printf("File: %s, Line: %d\n", __FILE__, __LINE__);

      return vftr_MPI_Ialltoallv(sendbuf, tmp_sendcounts, tmp_sdispls, sendtype,
                                 recvbuf, tmp_recvcounts, tmp_rdispls, recvtype,
                                 comm, request);
printf("File: %s, Line: %d\n", __FILE__, __LINE__);
   }
}

#endif
