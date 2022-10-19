#ifndef ISCATTER_H
#define ISCATTER_H

#include <mpi.h>

int vftr_MPI_Iscatter(const void *sendbuf, int sendcount,
                      MPI_Datatype sendtype, void *recvbuf,
                      int recvcount, MPI_Datatype recvtype,
                      int root, MPI_Comm comm,
                      MPI_Request *request);

int vftr_MPI_Iscatter_inplace(const void *sendbuf, int sendcount,
                              MPI_Datatype sendtype, void *recvbuf,
                              int recvcount, MPI_Datatype recvtype,
                              int root, MPI_Comm comm,
                              MPI_Request *request);

int vftr_MPI_Iscatter_intercom(const void *sendbuf, int sendcount,
                               MPI_Datatype sendtype, void *recvbuf,
                               int recvcount, MPI_Datatype recvtype,
                               int root, MPI_Comm comm,
                               MPI_Request *request);

#endif
