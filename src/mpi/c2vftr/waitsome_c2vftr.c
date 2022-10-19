#ifdef _MPI
#include <mpi.h>

#include "waitsome.h"

int vftr_MPI_Waitsome_c2vftr(int incount, MPI_Request array_of_requests[],
                             int *outcount, int array_of_indices[],
                             MPI_Status array_of_statuses[]) {
   return vftr_MPI_Waitsome(incount, array_of_requests, outcount,
                            array_of_indices, array_of_statuses);
}

#endif
