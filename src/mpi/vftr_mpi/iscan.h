#ifndef ISCAN_H
#define ISCAN_H

#include <mpi.h>

int vftr_MPI_Iscan(const void *sendbuf, void *recvbuf, int count,
                   MPI_Datatype datatype, MPI_Op op, MPI_Comm comm,
                   MPI_Request *request);

#endif
