#ifndef SCATTER_H
#define SCATTER_H

#include <mpi.h>

int vftr_MPI_Scatter(const void *sendbuf, int sendcount,
                     MPI_Datatype sendtype, void *recvbuf,
                     int recvcount, MPI_Datatype recvtype,
                     int root, MPI_Comm comm);

int vftr_MPI_Scatter_inplace(const void *sendbuf, int sendcount,
                             MPI_Datatype sendtype, void *recvbuf,
                             int recvcount, MPI_Datatype recvtype,
                             int root, MPI_Comm comm);

int vftr_MPI_Scatter_intercom(const void *sendbuf, int sendcount,
                              MPI_Datatype sendtype, void *recvbuf,
                              int recvcount, MPI_Datatype recvtype,
                              int root, MPI_Comm comm);

#endif
