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
#include "vftr_mpi_iallgatherv.h"

int MPI_Iallgatherv(const void *sendbuf, int sendcount,
                    MPI_Datatype sendtype, void *recvbuf,
                    const int *recvcounts, const int *displs,
                    MPI_Datatype recvtype, MPI_Comm comm,
                    MPI_Request *request) {
   if (vftr_no_mpi_logging()) {
      return PMPI_Iallgatherv(sendbuf, sendcount, sendtype, recvbuf,
                              recvcounts, displs, recvtype, comm,
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
      int *tmp_recvcounts = (int*) malloc(size*sizeof(int));
      for (int i=0; i<size; i++) {
         tmp_recvcounts[i] = recvcounts[i];
      }
      int *tmp_displs = (int*) malloc(size*sizeof(int));
      for (int i=0; i<size; i++) {
         tmp_displs[i] = displs[i];
      }

      return vftr_MPI_Iallgatherv(sendbuf, sendcount, sendtype, recvbuf,
                                  tmp_recvcounts, tmp_displs, recvtype, comm,
                                  request);
   }
}

#endif
