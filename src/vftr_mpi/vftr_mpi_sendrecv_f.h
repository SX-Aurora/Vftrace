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

#ifndef VFTR_MPI_SENDRECV_F_H
#define VFTR_MPI_SENDRECV_F_H

#ifdef _MPI
#include <mpi.h>

void vftr_MPI_Sendrecv_F(void *sendbuf, MPI_Fint *sendcount,
                         MPI_Fint *f_sendtype, MPI_Fint *dest, MPI_Fint *sendtag,
                         void *recvbuf, MPI_Fint *recvcount, MPI_Fint *f_recvtype,
                         MPI_Fint *source, MPI_Fint *recvtag, MPI_Fint *f_comm,
                         MPI_Fint *f_status, MPI_Fint *f_error);

#endif
#endif
