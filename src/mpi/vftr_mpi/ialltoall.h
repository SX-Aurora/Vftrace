#ifndef IALLTOALL_H
#define IALLTOALL_H

#include <mpi.h>

int vftr_MPI_Ialltoall(const void *sendbuf, int sendcount,
                       MPI_Datatype sendtype, void *recvbuf,
                       int recvcount, MPI_Datatype recvtype,
                       MPI_Comm comm, MPI_Request *request);

int vftr_MPI_Ialltoall_inplace(const void *sendbuf, int sendcount,
                               MPI_Datatype sendtype, void *recvbuf,
                               int recvcount, MPI_Datatype recvtype,
                               MPI_Comm comm, MPI_Request *request);

int vftr_MPI_Ialltoall_intercom(const void *sendbuf, int sendcount,
                                MPI_Datatype sendtype, void *recvbuf,
                                int recvcount, MPI_Datatype recvtype,
                                MPI_Comm comm, MPI_Request *request);

#endif
