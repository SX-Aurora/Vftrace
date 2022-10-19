#ifndef RSEND_H
#define RSEND_H

#include <mpi.h>

int vftr_MPI_Rsend(const void *buf, int count, MPI_Datatype datatype,
                   int dest, int tag, MPI_Comm comm);

#endif
