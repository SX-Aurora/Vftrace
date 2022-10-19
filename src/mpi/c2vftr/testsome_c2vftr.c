#ifdef _MPI
#include <mpi.h>

#include "testsome.h"

int vftr_MPI_Testsome_c2vftr(int incount, MPI_Request array_of_requests[],
                             int *outcount, int array_of_indices[],
                             MPI_Status array_of_statuses[]) {
   return vftr_MPI_Testsome(incount, array_of_requests, outcount,
                            array_of_indices, array_of_statuses);
}

#endif
