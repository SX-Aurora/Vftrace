#ifndef SEND_H
#define SEND_H

#include <mpi.h>

int vftr_MPI_Send(const void *buf, int count, MPI_Datatype datatype,
                  int dest, int tag, MPI_Comm comm);

#endif
