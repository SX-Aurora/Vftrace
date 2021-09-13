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

#ifndef IREDUCE_SCATTER_BLOCK_H
#define IREDUCE_SCATTER_BLOCK_H

#ifdef _MPI
#include <mpi.h>

int vftr_MPI_Ireduce_scatter_block(const void *sendbuf, void *recvbuf,
                                   int recvcount, MPI_Datatype datatype,
                                   MPI_Op op, MPI_Comm comm,
                                   MPI_Request *request);

int vftr_MPI_Ireduce_scatter_block_inplace(const void *sendbuf, void *recvbuf,
                                           int recvcount, MPI_Datatype datatype,
                                           MPI_Op op, MPI_Comm comm,
                                           MPI_Request *request);

int vftr_MPI_Ireduce_scatter_block_intercom(const void *sendbuf, void *recvbuf,
                                            int recvcount, MPI_Datatype datatype,
                                            MPI_Op op, MPI_Comm comm,
                                            MPI_Request *request);

#endif
#endif
