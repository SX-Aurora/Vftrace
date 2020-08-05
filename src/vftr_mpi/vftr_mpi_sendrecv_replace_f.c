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

#include "vftr_mpi_sendrecv_replace.h"

void vftr_MPI_Sendrecv_replace_F(void *buf, MPI_Fint *count, MPI_Fint *f_datatype,
                                 MPI_Fint *dest, MPI_Fint *sendtag, MPI_Fint *source,
                                 MPI_Fint *recvtag, MPI_Fint *f_comm, MPI_Fint *f_status,
                                 MPI_Fint *f_error) {

   MPI_Comm c_comm = PMPI_Comm_f2c(*f_comm);
   MPI_Datatype c_datatype = PMPI_Type_f2c(*f_datatype);
   MPI_Status c_status;

   int c_error = vftr_MPI_Sendrecv_replace(buf,
                                           (int)(*count),
                                           c_datatype,
                                           (int)(*dest),
                                           (int)(*sendtag),
                                           (int)(*source),
                                           (int)(*recvtag),
                                           c_comm,
                                           &c_status);

   if (f_status != MPI_F_STATUS_IGNORE) {
      MPI_Status_c2f(&c_status, f_status);
   }
   *f_error = (MPI_Fint) (c_error);
}

#endif
