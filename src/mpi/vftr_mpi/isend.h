#ifndef ISEND_H
#define ISEND_H

#include <mpi.h>

int vftr_MPI_Isend(const void *buf, int count, MPI_Datatype datatype,
                   int dest, int tag, MPI_Comm comm,
                   MPI_Request *request);

#endif
