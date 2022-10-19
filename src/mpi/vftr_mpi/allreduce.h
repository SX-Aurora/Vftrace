#ifndef ALLREDUCE_H
#define ALLREDUCE_H

#include <mpi.h>

int vftr_MPI_Allreduce(const void *sendbuf, void *recvbuf, int count,
                       MPI_Datatype datatype, MPI_Op op, MPI_Comm comm);

int vftr_MPI_Allreduce_inplace(const void *sendbuf, void *recvbuf, int count,
                               MPI_Datatype datatype, MPI_Op op, MPI_Comm comm);

int vftr_MPI_Allreduce_intercom(const void *sendbuf, void *recvbuf, int count,
                                MPI_Datatype datatype, MPI_Op op, MPI_Comm comm);

#endif
