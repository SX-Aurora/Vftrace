#ifndef WAITANY_H
#define WAITANY_H

#include <mpi.h>

int vftr_MPI_Waitany(int count, MPI_Request array_of_requests[],
                     int *index, MPI_Status *status);

#endif
