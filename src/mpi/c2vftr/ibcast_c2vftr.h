#ifndef IBCAST_C2VFTR_H
#define IBCAST_C2VFTR_H

#ifdef _MPI
#include <mpi.h>

int vftr_MPI_Ibcast_c2vftr(void *buffer, int count, MPI_Datatype datatype,
                           int root, MPI_Comm comm, MPI_Request *request);

#endif
#endif
