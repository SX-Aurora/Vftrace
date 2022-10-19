#ifndef IREDUCE_H
#define IREDUCE_H

#include <mpi.h>

int vftr_MPI_Ireduce(const void *sendbuf, void *recvbuf, int count,
                     MPI_Datatype datatype, MPI_Op op, int root, MPI_Comm comm,
                     MPI_Request *request);

int vftr_MPI_Ireduce_inplace(const void *sendbuf, void *recvbuf, int count,
                             MPI_Datatype datatype, MPI_Op op, int root,
                             MPI_Comm comm, MPI_Request *request);

int vftr_MPI_Ireduce_intercom(const void *sendbuf, void *recvbuf, int count,
                              MPI_Datatype datatype, MPI_Op op, int root,
                              MPI_Comm comm, MPI_Request *request);

#endif
