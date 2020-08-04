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

#ifndef VFTR_MPI_POINT2POINT_H
#define VFTR_MPI_POINT2POINT_H

#ifdef _MPI
#include <mpi.h>

int vftr_MPI_Send(const void *buf, int count, MPI_Datatype datatype,
                  int dest, int tag, MPI_Comm comm);

int vftr_MPI_Bsend(const void *buf, int count, MPI_Datatype datatype,
                   int dest, int tag, MPI_Comm comm);
   
int vftr_MPI_Bsend_init(const void *buf, int count,
                        MPI_Datatype datatype, int dest, int tag,
                        MPI_Comm comm, MPI_Request *request);

int vftr_MPI_Isend(const void *buf, int count, MPI_Datatype datatype,
                   int dest, int tag, MPI_Comm comm,
                   MPI_Request *request);

int vftr_MPI_Ibsend(const void *buf, int count, MPI_Datatype datatype,
                    int dest, int tag, MPI_Comm comm, MPI_Request *request);

int vftr_MPI_Ssend(const void *buf, int count, MPI_Datatype datatype,
                   int dest, int tag, MPI_Comm comm);

int vftr_MPI_Issend(const void *buf, int count, MPI_Datatype datatype,
                    int dest, int tag, MPI_Comm comm, MPI_Request *request);
 
int vftr_MPI_Rsend(const void *buf, int count, MPI_Datatype datatype,
                   int dest, int tag, MPI_Comm comm);

int vftr_MPI_Irsend(const void *buf, int count, MPI_Datatype datatype,
                    int dest, int tag, MPI_Comm comm, MPI_Request *request);

int vftr_MPI_Recv(void *buf, int count, MPI_Datatype datatype,
                  int source, int tag, MPI_Comm comm, MPI_Status *status);

int vftr_MPI_Irecv(void *buf, int count, MPI_Datatype datatype,
                   int source, int tag, MPI_Comm comm, MPI_Request *request);

int vftr_MPI_Sendrecv(const void *sendbuf, int sendcount,
                      MPI_Datatype sendtype, int dest, int sendtag,
                      void *recvbuf, int recvcount, MPI_Datatype recvtype,
                      int source, int recvtag, MPI_Comm comm,
                      MPI_Status *status);

int vftr_MPI_Sendrecv_replace(void *buf, int count, MPI_Datatype datatype,
                              int dest, int sendtag, int source, int recvtag,
                              MPI_Comm comm, MPI_Status *status);

#endif
#endif
