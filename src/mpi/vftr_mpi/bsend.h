#ifndef BSEND_H
#define BSEND_H

#include <mpi.h>

int vftr_MPI_Bsend(const void *buf, int count, MPI_Datatype datatype,
                   int dest, int tag, MPI_Comm comm);

#endif
