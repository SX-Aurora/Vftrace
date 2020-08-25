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

#include "vftr_mpi_buf_addr_const.h"
#include "vftr_mpi_alltoallw.h"

void vftr_MPI_Alltoallw_F(void *sendbuf, MPI_Fint *f_sendcounts, MPI_Fint *f_sdispls,
                          MPI_Fint *f_sendtypes, void *recvbuf, MPI_Fint *f_recvcounts,
                          MPI_Fint *f_rdispls, MPI_Fint *f_recvtypes,
                          MPI_Fint *f_comm, MPI_Fint *f_error) {

   MPI_Comm c_comm = PMPI_Comm_f2c(*f_comm);

   int size;
   int isintercom;
   PMPI_Comm_test_inter(c_comm, &isintercom);
   if (isintercom) {
      PMPI_Comm_remote_size(c_comm, &size);
   } else {
      PMPI_Comm_size(c_comm, &size);
   }

   int *c_sendcounts = (int*) malloc(size*sizeof(int));
   for (int i=0; i<size; i++) {
      c_sendcounts[i] = (int) f_sendcounts[i];
   }
   int *c_sdispls = (int*) malloc(size*sizeof(int));
   for (int i=0; i<size; i++) {
      c_sdispls[i] = (int) f_sdispls[i];
   }
   MPI_Datatype *c_sendtypes = (MPI_Datatype*) malloc(size*sizeof(MPI_Datatype));
   for (int i=0; i<size; i++) {
      c_sendtypes[i] = PMPI_Type_f2c(f_sendtypes[i]);
   }

   int *c_recvcounts = (int*) malloc(size*sizeof(int));
   for (int i=0; i<size; i++) {
      c_recvcounts[i] = (int) f_recvcounts[i];
   }
   int *c_rdispls = (int*) malloc(size*sizeof(int));
   for (int i=0; i<size; i++) {
      c_rdispls[i] = (int) f_rdispls[i];
   }
   MPI_Datatype *c_recvtypes = (MPI_Datatype*) malloc(size*sizeof(MPI_Datatype));
   for (int i=0; i<size; i++) {
      c_recvtypes[i] = PMPI_Type_f2c(f_recvtypes[i]);
   }

   sendbuf = (void*) vftr_is_F_MPI_IN_PLACE(sendbuf) ? MPI_IN_PLACE : sendbuf;
   sendbuf = (void*) vftr_is_F_MPI_BOTTOM(sendbuf) ? MPI_BOTTOM : sendbuf;
   recvbuf = (void*) vftr_is_F_MPI_BOTTOM(recvbuf) ? MPI_BOTTOM : recvbuf;

   int c_error = vftr_MPI_Alltoallw(sendbuf,
                                    c_sendcounts,
                                    c_sdispls,
                                    c_sendtypes,
                                    recvbuf,
                                    c_recvcounts,
                                    c_rdispls,
                                    c_recvtypes,
                                    c_comm);

   free(c_sendcounts);
   free(c_sdispls);
   free(c_sendtypes);
   free(c_recvcounts);
   free(c_rdispls);
   free(c_recvtypes);

   *f_error = (MPI_Fint) (c_error);
}

#endif
