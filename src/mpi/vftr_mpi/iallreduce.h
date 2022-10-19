#ifndef IALLREDUCE_H
#define IALLREDUCE_H

#include <mpi.h>

int vftr_MPI_Iallreduce(const void *sendbuf, void *recvbuf, int count,
                        MPI_Datatype datatype, MPI_Op op, MPI_Comm comm,
                        MPI_Request *request);

int vftr_MPI_Iallreduce_inplace(const void *sendbuf, void *recvbuf, int count,
                                MPI_Datatype datatype, MPI_Op op, MPI_Comm comm,
                                MPI_Request *request);

int vftr_MPI_Iallreduce_intercom(const void *sendbuf, void *recvbuf, int count,
                                 MPI_Datatype datatype, MPI_Op op, MPI_Comm comm,
                                 MPI_Request *request);

#endif
