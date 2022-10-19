#ifndef WAITSOME_C2VFTR_H
#define WAITSOME_C2VFTR_H

#ifdef _MPI
#include <mpi.h>

int vftr_MPI_Waitsome_c2vftr(int incount, MPI_Request array_of_requests[],
                             int *outcount, int array_of_indices[],
                             MPI_Status array_of_statuses[]);

#endif
#endif
