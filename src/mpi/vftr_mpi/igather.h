#ifndef IGATHER_H
#define IGATHER_H

#include <mpi.h>

int vftr_MPI_Igather(const void *sendbuf, int sendcount,
                     MPI_Datatype sendtype, void *recvbuf,
                     int recvcount, MPI_Datatype recvtype,
                     int root, MPI_Comm comm,
                     MPI_Request *request);

int vftr_MPI_Igather_inplace(const void *sendbuf, int sendcount,
                             MPI_Datatype sendtype, void *recvbuf,
                             int recvcount, MPI_Datatype recvtype,
                             int root, MPI_Comm comm,
                             MPI_Request *request);

int vftr_MPI_Igather_intercom(const void *sendbuf, int sendcount,
                              MPI_Datatype sendtype, void *recvbuf,
                              int recvcount, MPI_Datatype recvtype,
                              int root, MPI_Comm comm,
                              MPI_Request *request);

#endif
