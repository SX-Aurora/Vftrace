#ifndef ALLTOALL_H
#define ALLTOALL_H

#include <mpi.h>

int vftr_MPI_Alltoall(const void *sendbuf, int sendcount,
                      MPI_Datatype sendtype, void *recvbuf,
                      int recvcount, MPI_Datatype recvtype,
                      MPI_Comm comm);

int vftr_MPI_Alltoall_inplace(const void *sendbuf, int sendcount,
                              MPI_Datatype sendtype, void *recvbuf,
                              int recvcount, MPI_Datatype recvtype,
                              MPI_Comm comm);

int vftr_MPI_Alltoall_intercom(const void *sendbuf, int sendcount,
                               MPI_Datatype sendtype, void *recvbuf,
                               int recvcount, MPI_Datatype recvtype,
                               MPI_Comm comm);

#endif
