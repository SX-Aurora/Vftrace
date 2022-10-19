#ifndef TESTANY_H
#define TESTANY_H

#include <mpi.h>

int vftr_MPI_Testany(int count, MPI_Request array_of_requests[],
                     int *index, int *flag, MPI_Status *status);

#endif
