#ifndef TESTSOME_H
#define TESTSOME_H

#include <mpi.h>

int vftr_MPI_Testsome(int incount, MPI_Request array_of_requests[],
                      int *outcount, int array_of_indices[],
                      MPI_Status array_of_statuses[]);

#endif
