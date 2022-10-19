#ifndef GATHERV_H
#define GATHERV_H

#include <mpi.h>

int vftr_MPI_Gatherv(const void *sendbuf, int sendcount,
                     MPI_Datatype sendtype, void *recvbuf,
                     const int *recvcounts, const int *displs,
                     MPI_Datatype recvtype, int root,
                     MPI_Comm comm);

int vftr_MPI_Gatherv_inplace(const void *sendbuf, int sendcount,
                             MPI_Datatype sendtype, void *recvbuf,
                             const int *recvcounts, const int *displs,
                             MPI_Datatype recvtype, int root,
                             MPI_Comm comm);

int vftr_MPI_Gatherv_intercom(const void *sendbuf, int sendcount,
                              MPI_Datatype sendtype, void *recvbuf,
                              const int *recvcounts, const int *displs,
                              MPI_Datatype recvtype, int root,
                              MPI_Comm comm);

#endif
