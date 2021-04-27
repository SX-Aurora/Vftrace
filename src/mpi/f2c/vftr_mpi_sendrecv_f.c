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

#include "vftr_mpi_sendrecv.h"

void vftr_MPI_Sendrecv_F(void *sendbuf, MPI_Fint *sendcount,
                         MPI_Fint *f_sendtype, MPI_Fint *dest, MPI_Fint *sendtag,
                         void *recvbuf, MPI_Fint *recvcount, MPI_Fint *f_recvtype,
                         MPI_Fint *source, MPI_Fint *recvtag, MPI_Fint *f_comm,
                         MPI_Fint *f_status, MPI_Fint *f_error) {

   MPI_Comm c_comm = PMPI_Comm_f2c(*f_comm);
   MPI_Datatype c_sendtype = PMPI_Type_f2c(*f_sendtype);
   MPI_Datatype c_recvtype = PMPI_Type_f2c(*f_recvtype);
   MPI_Status c_status;

   int c_error = vftr_MPI_Sendrecv(sendbuf,
                                   (int)(*sendcount),
                                   c_sendtype,
                                   (int)(*dest),
                                   (int)(*sendtag),
                                   recvbuf,
                                   (int)(*recvcount),
                                   c_recvtype,
                                   (int)(*source),
                                   (int)(*recvtag),
                                   c_comm,
                                   &c_status);

   if (f_status != MPI_F_STATUS_IGNORE) {
      PMPI_Status_c2f(&c_status, f_status);
   }
   *f_error = (MPI_Fint) (c_error);
}

#endif
