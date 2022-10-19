#ifndef BCAST_C2VFTR_H
#define BCAST_C2VFTR_H

#ifdef _MPI
#include <mpi.h>

int vftr_MPI_Bcast_c2vftr(void *buffer, int count, MPI_Datatype datatype,
                          int root, MPI_Comm comm);

#endif
#endif
