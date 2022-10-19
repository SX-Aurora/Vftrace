#ifndef REDUCE_H
#define REDUCE_H

#include <mpi.h>

int vftr_MPI_Reduce(const void *sendbuf, void *recvbuf, int count,
                    MPI_Datatype datatype, MPI_Op op, int root, MPI_Comm comm);

int vftr_MPI_Reduce_inplace(const void *sendbuf, void *recvbuf, int count,
                            MPI_Datatype datatype, MPI_Op op, int root,
                            MPI_Comm comm);

int vftr_MPI_Reduce_intercom(const void *sendbuf, void *recvbuf, int count,
                             MPI_Datatype datatype, MPI_Op op, int root,
                             MPI_Comm comm);

#endif
