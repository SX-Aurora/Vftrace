#ifndef BCAST_H
#define BCAST_H

#include <mpi.h>

int vftr_MPI_Bcast(void *buffer, int count, MPI_Datatype datatype,
                   int root, MPI_Comm comm);

int vftr_MPI_Bcast_intercom(void *buffer, int count, MPI_Datatype datatype,
                            int root, MPI_Comm comm);

#endif
