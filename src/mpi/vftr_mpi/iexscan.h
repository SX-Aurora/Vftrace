#ifndef IEXSCAN_H
#define IEXSCAN_H

#include <mpi.h>

int vftr_MPI_Iexscan(const void *sendbuf, void *recvbuf, int count,
                     MPI_Datatype datatype, MPI_Op op, MPI_Comm comm,
                     MPI_Request *request);

#endif
