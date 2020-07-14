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

#include "vftr_mpi_collective.h"

int MPI_Allgather(const void *sendbuf, int sendcount,
                  MPI_Datatype sendtype, void *recvbuf, int recvcount,
                  MPI_Datatype recvtype, MPI_Comm comm) {
   return vftr_MPI_Allgather(sendbuf, sendcount, sendtype, recvbuf,
                            recvcount, recvtype, comm);
}

int MPI_Allgatherv(const void *sendbuf, int sendcount,
                   MPI_Datatype sendtype, void *recvbuf,
                   const int *recvcounts, const int *displs,
                   MPI_Datatype recvtype, MPI_Comm comm) {
   return vftr_MPI_Allgatherv(sendbuf, sendcount, sendtype, recvbuf,
                              recvcounts, displs, recvtype, comm);
}

int MPI_Allreduce(const void *sendbuf, void *recvbuf, int count,
                  MPI_Datatype datatype, MPI_Op op, MPI_Comm comm) {
   return vftr_MPI_Allreduce(sendbuf, recvbuf, count, datatype, op, comm);
}

int MPI_Alltoall(const void *sendbuf, int sendcount,
                 MPI_Datatype sendtype, void *recvbuf, int recvcount,
                 MPI_Datatype recvtype, MPI_Comm comm) {
   return vftr_MPI_Alltoall(sendbuf, sendcount, sendtype, recvbuf,
                            recvcount, recvtype, comm);
}

int MPI_Alltoallv(const void *sendbuf, const int *sendcounts,
                  const int *sdispls, MPI_Datatype sendtype, void *recvbuf,
                  const int *recvcounts, const int *rdispls,
                  MPI_Datatype recvtype, MPI_Comm comm) {
   return vftr_MPI_Alltoallv(sendbuf, sendcounts, sdispls, sendtype,
                             recvbuf, recvcounts, rdispls, recvtype, comm);
}

int MPI_Alltoallw(const void *sendbuf, const int *sendcounts,
                  const int *sdispls, const MPI_Datatype *sendtypes,
                  void *recvbuf, const int *recvcounts, const int *rdispls,
                  const MPI_Datatype *recvtypes, MPI_Comm comm) {
   return vftr_MPI_Alltoallw(sendbuf, sendcounts, sdispls, sendtypes,
                             recvbuf, recvcounts, rdispls, recvtypes, comm);
}

int MPI_Bcast(void *buffer, int count, MPI_Datatype datatype,
              int root, MPI_Comm comm) {
   return vftr_MPI_Bcast(buffer, count, datatype, root, comm);
}

int MPI_Gather(const void *sendbuf, int sendcount,
               MPI_Datatype sendtype, void *recvbuf, int recvcount,
               MPI_Datatype recvtype, int root, MPI_Comm comm) {
   return vftr_MPI_Gather(sendbuf, sendcount, sendtype, recvbuf, recvcount,
                          recvtype, root, comm);
}

int MPI_Gatherv(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
                void *recvbuf, const int *recvcounts, const int *displs,
                MPI_Datatype recvtype, int root, MPI_Comm comm) {
   return vftr_MPI_Gatherv(sendbuf, sendcount, sendtype, recvbuf,
                           recvcounts, displs, recvtype, root, comm);
}

int MPI_Reduce(const void *sendbuf, void *recvbuf, int count,
               MPI_Datatype datatype, MPI_Op op, int root, MPI_Comm comm) {
   return vftr_MPI_Reduce(sendbuf, recvbuf, count, datatype, op, root, comm);
}

int MPI_Reduce_scatter(const void *sendbuf, void *recvbuf,
                       const int *recvcounts, MPI_Datatype datatype,
                       MPI_Op op, MPI_Comm comm) {
   return vftr_MPI_Reduce_scatter(sendbuf, recvbuf, recvcounts, datatype, op, comm);
}

int MPI_Scatter(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
                void *recvbuf, int recvcount, MPI_Datatype recvtype,
                int root, MPI_Comm comm) {
   return vftr_MPI_Scatter(sendbuf, sendcount, sendtype, recvbuf, recvcount,
                           recvtype, root, comm);
}

int MPI_Scatterv(const void *sendbuf, const int *sendcounts,
                 const int *displs, MPI_Datatype sendtype,
                 void *recvbuf, int recvcount, MPI_Datatype recvtype,
                 int root, MPI_Comm comm) {
   return vftr_MPI_Scatterv(sendbuf, sendcounts, displs, sendtype,
                            recvbuf, recvcount, recvtype, root, comm);
}

#endif
