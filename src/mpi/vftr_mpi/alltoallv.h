#ifndef ALLTOALLV_H
#define ALLTOALLV_H

#include <mpi.h>

int vftr_MPI_Alltoallv(const void *sendbuf, const int *sendcounts,
                       const int *sdispls, MPI_Datatype sendtype,
                       void *recvbuf, const int *recvcounts,
                       const int *rdispls, MPI_Datatype recvtype,
                       MPI_Comm comm);

int vftr_MPI_Alltoallv_inplace(const void *sendbuf, const int *sendcounts,
                               const int *sdispls, MPI_Datatype sendtype,
                               void *recvbuf, const int *recvcounts,
                               const int *rdispls, MPI_Datatype recvtype,
                               MPI_Comm comm);

int vftr_MPI_Alltoallv_intercom(const void *sendbuf, const int *sendcounts,
                                const int *sdispls, MPI_Datatype sendtype,
                                void *recvbuf, const int *recvcounts,
                                const int *rdispls, MPI_Datatype recvtype,
                                MPI_Comm comm);

#endif
