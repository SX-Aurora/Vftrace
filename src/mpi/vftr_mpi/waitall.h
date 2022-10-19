#ifndef WAITALL_H
#define WAITALL_H

#include <mpi.h>

int vftr_MPI_Waitall(int count, MPI_Request array_of_requests[],
                     MPI_Status array_of_statuses[]);

#endif
