#ifndef RECV_H
#define RECV_H

#include <mpi.h>

int vftr_MPI_Recv(void *buf, int count, MPI_Datatype datatype,
                  int source, int tag, MPI_Comm comm, MPI_Status *status);

#endif
