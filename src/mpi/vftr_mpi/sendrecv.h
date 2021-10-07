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

#ifndef SENDRECV_H
#define SENDRECV_H

#include <mpi.h>

int vftr_MPI_Sendrecv(const void *sendbuf, int sendcount,
                      MPI_Datatype sendtype, int dest, int sendtag,
                      void *recvbuf, int recvcount, MPI_Datatype recvtype,
                      int source, int recvtag, MPI_Comm comm,
                      MPI_Status *status);

#endif
