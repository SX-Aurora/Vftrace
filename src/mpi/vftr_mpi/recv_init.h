#ifndef RECV_INIT_H
#define RECV_INIT_H

#include <mpi.h>

int vftr_MPI_Recv_init(void *buf, int count, MPI_Datatype datatype,
                       int source, int tag, MPI_Comm comm, MPI_Request *request);

#endif
