#ifndef IALLTOALLV_H
#define IALLTOALLV_H

#include <mpi.h>

int vftr_MPI_Ialltoallv(const void *sendbuf, const int *sendcounts,
                        const int *sdispls, MPI_Datatype sendtype,
                        void *recvbuf, const int *recvcounts,
                        const int *rdispls, MPI_Datatype recvtype,
                        MPI_Comm comm, MPI_Request *request);

int vftr_MPI_Ialltoallv_inplace(const void *sendbuf, const int *sendcounts,
                                const int *sdispls, MPI_Datatype sendtype,
                                void *recvbuf, const int *recvcounts,
                                const int *rdispls, MPI_Datatype recvtype,
                                MPI_Comm comm, MPI_Request *request);

int vftr_MPI_Ialltoallv_intercom(const void *sendbuf, const int *sendcounts,
                                 const int *sdispls, MPI_Datatype sendtype,
                                 void *recvbuf, const int *recvcounts,
                                 const int *rdispls, MPI_Datatype recvtype,
                                 MPI_Comm comm, MPI_Request *request);

#endif
