#ifndef ISCATTERV_H
#define ISCATTERV_H

#include <mpi.h>

int vftr_MPI_Iscatterv(const void *sendbuf, const int *sendcounts,
                       const int *displs, MPI_Datatype sendtype,
                       void *recvbuf, int recvcount,
                       MPI_Datatype recvtype, int root,
                       MPI_Comm comm, MPI_Request *request);

int vftr_MPI_Iscatterv_inplace(const void *sendbuf, const int *sendcounts,
                               const int *displs, MPI_Datatype sendtype,
                               void *recvbuf, int recvcount,
                               MPI_Datatype recvtype, int root,
                               MPI_Comm comm, MPI_Request *request);

int vftr_MPI_Iscatterv_intercom(const void *sendbuf, const int *sendcounts,
                                const int *displs, MPI_Datatype sendtype,
                                void *recvbuf, int recvcount,
                                MPI_Datatype recvtype, int root,
                                MPI_Comm comm, MPI_Request *request);

#endif
