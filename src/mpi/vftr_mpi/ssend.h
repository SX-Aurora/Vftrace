#ifndef SSEND_H
#define SSEND_H

#include <mpi.h>

int vftr_MPI_Ssend(const void *buf, int count, MPI_Datatype datatype,
                   int dest, int tag, MPI_Comm comm);

#endif
