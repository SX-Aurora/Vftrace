#ifndef IRECV_H
#define IRECV_H

#include <mpi.h>

int vftr_MPI_Irecv(void *buf, int count, MPI_Datatype datatype,
                   int source, int tag, MPI_Comm comm, MPI_Request *request);

#endif
