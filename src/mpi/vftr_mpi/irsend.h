#ifndef IRSEND_H
#define IRSEND_H

#include <mpi.h>

int vftr_MPI_Irsend(const void *buf, int count, MPI_Datatype datatype,
                    int dest, int tag, MPI_Comm comm,
                    MPI_Request *request);

#endif
