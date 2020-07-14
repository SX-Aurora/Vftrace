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

#include "vftr_mpi_point2point.h"

int MPI_Send(const void *buf, int count, MPI_Datatype datatype,
             int dest, int tag, MPI_Comm comm) {
   return vftr_MPI_Send(buf, count, datatype, dest, tag, comm);
}

int MPI_Bsend(const void *buf, int count, MPI_Datatype datatype,
              int dest, int tag, MPI_Comm comm) {
   return vftr_MPI_Bsend(buf, count, datatype, dest, tag, comm);
}

int MPI_Bsend_init(const void *buf, int count,
                   MPI_Datatype datatype, int dest, int tag,
                   MPI_Comm comm, MPI_Request *request) {
   return vftr_MPI_Bsend_init(buf, count, datatype, dest, tag, comm, request);
}

int MPI_Isend(const void *buf, int count, MPI_Datatype datatype,
              int dest, int tag, MPI_Comm comm,
              MPI_Request *request) {
   return vftr_MPI_Isend(buf, count, datatype, dest, tag, comm, request);
}

int MPI_Ibsend(const void *buf, int count, MPI_Datatype datatype,
               int dest, int tag, MPI_Comm comm, MPI_Request *request) {
   return vftr_MPI_Ibsend(buf, count, datatype, dest, tag, comm, request);
}

int MPI_Ssend(const void *buf, int count, MPI_Datatype datatype,
              int dest, int tag, MPI_Comm comm) {
   return vftr_MPI_Ssend(buf, count, datatype, dest, tag, comm);
}

int MPI_Issend(const void *buf, int count, MPI_Datatype datatype,
               int dest, int tag, MPI_Comm comm, MPI_Request *request) {
   return vftr_MPI_Issend(buf, count, datatype, dest, tag, comm, request);
}

int MPI_Rsend(const void *buf, int count, MPI_Datatype datatype,
              int dest, int tag, MPI_Comm comm) {
   return vftr_MPI_Rsend(buf, count, datatype, dest, tag, comm);
}

int MPI_Irsend(const void *buf, int count, MPI_Datatype datatype,
               int dest, int tag, MPI_Comm comm, MPI_Request *request) {
   return vftr_MPI_Irsend(buf, count, datatype, dest, tag, comm, request);
}

int MPI_Recv(void *buf, int count, MPI_Datatype datatype,
             int source, int tag, MPI_Comm comm, MPI_Status *status) {
   return vftr_MPI_Recv(buf, count, datatype, source, tag, comm, status);
}

int MPI_Irecv(void *buf, int count, MPI_Datatype datatype,
              int source, int tag, MPI_Comm comm, MPI_Request *request) {
   return vftr_MPI_Irecv(buf, count, datatype, source, tag, comm, request);
}

int MPI_Sendrecv(const void *sendbuf, int sendcount,
                 MPI_Datatype sendtype, int dest, int sendtag,
                 void *recvbuf, int recvcount, MPI_Datatype recvtype,
                 int source, int recvtag, MPI_Comm comm,
                 MPI_Status *status) {
   return vftr_MPI_Sendrecv(sendbuf, sendcount, sendtype, dest, sendtag,
                            recvbuf, recvcount, recvtype, source, recvtag,
                            comm, status);
}

int MPI_Sendrecv_replace(void *buf, int count, MPI_Datatype datatype,
                         int dest, int sendtag, int source, int recvtag,
                         MPI_Comm comm, MPI_Status *status) {
   return vftr_MPI_Sendrecv_replace(buf, count, datatype, dest, sendtag,
                                    source, recvtag, comm, status);
}

#endif
