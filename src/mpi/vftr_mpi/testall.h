#ifndef TESTALL_H
#define TESTALL_H

#include <mpi.h>

int vftr_MPI_Testall(int count, MPI_Request array_of_requests[],
                     int *flag, MPI_Status array_of_statuses[]);

#endif
