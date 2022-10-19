#ifndef IALLGATHERV_H
#define IALLGATHERV_H

#include <mpi.h>

int vftr_MPI_Iallgatherv(const void *sendbuf, int sendcount,
                         MPI_Datatype sendtype, void *recvbuf,
                         const int *recvcounts, const int *displs,
                         MPI_Datatype recvtype, MPI_Comm comm,
                         MPI_Request *request);

int vftr_MPI_Iallgatherv_inplace(const void *sendbuf, int sendcount,
                                 MPI_Datatype sendtype, void *recvbuf,
                                 const int *recvcounts, const int *displs,
                                 MPI_Datatype recvtype, MPI_Comm comm,
                                 MPI_Request *request);

int vftr_MPI_Iallgatherv_intercom(const void *sendbuf, int sendcount,
                                  MPI_Datatype sendtype, void *recvbuf,
                                  const int *recvcounts, const int *displs,
                                  MPI_Datatype recvtype, MPI_Comm comm,
                                  MPI_Request *request);

#endif
