#ifndef ALLGATHER_H
#define ALLGATHER_H

#include <mpi.h>

int vftr_MPI_Allgather(const void *sendbuf, int sendcount,
                       MPI_Datatype sendtype, void *recvbuf,
                       int recvcount, MPI_Datatype recvtype,
                       MPI_Comm comm);

int vftr_MPI_Allgather_inplace(const void *sendbuf, int sendcount,
                               MPI_Datatype sendtype, void *recvbuf,
                               int recvcount, MPI_Datatype recvtype,
                               MPI_Comm comm);

int vftr_MPI_Allgather_intercom(const void *sendbuf, int sendcount,
                                MPI_Datatype sendtype, void *recvbuf,
                                int recvcount, MPI_Datatype recvtype,
                                MPI_Comm comm);

#endif
