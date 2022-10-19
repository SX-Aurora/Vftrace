#ifndef IBARRIER_H
#define IBARRIER_H

#include <mpi.h>

int vftr_MPI_Ibarrier(MPI_Comm comm, MPI_Request *request);

#endif
