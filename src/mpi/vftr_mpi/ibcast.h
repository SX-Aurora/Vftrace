#ifndef IBCAST_H
#define IBCAST_H

#include <mpi.h>

int vftr_MPI_Ibcast(void *buffer, int count, MPI_Datatype datatype,
                    int root, MPI_Comm comm, MPI_Request *request);

int vftr_MPI_Ibcast_intercom(void *buffer, int count, MPI_Datatype datatype,
                             int root, MPI_Comm comm, MPI_Request *request);

#endif
