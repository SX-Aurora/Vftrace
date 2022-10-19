#ifndef IGATHERV_H
#define IGATHERV_H

#include <mpi.h>

int vftr_MPI_Igatherv(const void *sendbuf, int sendcount,
                      MPI_Datatype sendtype, void *recvbuf,
                      const int *recvcounts, const int *displs,
                      MPI_Datatype recvtype, int root,
                      MPI_Comm comm, MPI_Request *request);

int vftr_MPI_Igatherv_inplace(const void *sendbuf, int sendcount,
                              MPI_Datatype sendtype, void *recvbuf,
                              const int *recvcounts, const int *displs,
                              MPI_Datatype recvtype, int root,
                              MPI_Comm comm, MPI_Request *request);

int vftr_MPI_Igatherv_intercom(const void *sendbuf, int sendcount,
                               MPI_Datatype sendtype, void *recvbuf,
                               const int *recvcounts, const int *displs,
                               MPI_Datatype recvtype, int root,
                               MPI_Comm comm, MPI_Request *request);

#endif
