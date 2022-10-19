#ifndef GATHER_H
#define GATHER_H

#include <mpi.h>

int vftr_MPI_Gather(const void *sendbuf, int sendcount,
                    MPI_Datatype sendtype, void *recvbuf,
                    int recvcount, MPI_Datatype recvtype,
                    int root, MPI_Comm comm);

int vftr_MPI_Gather_inplace(const void *sendbuf, int sendcount,
                            MPI_Datatype sendtype, void *recvbuf,
                            int recvcount, MPI_Datatype recvtype,
                            int root, MPI_Comm comm);

int vftr_MPI_Gather_intercom(const void *sendbuf, int sendcount,
                             MPI_Datatype sendtype, void *recvbuf,
                             int recvcount, MPI_Datatype recvtype,
                             int root, MPI_Comm comm);

#endif
