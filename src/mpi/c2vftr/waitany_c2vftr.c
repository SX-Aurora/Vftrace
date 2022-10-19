#ifdef _MPI
#include <mpi.h>

#include "waitany.h"

int vftr_MPI_Waitany_c2vftr(int count, MPI_Request array_of_requests[],
                            int *index, MPI_Status *status) {
   return vftr_MPI_Waitany(count, array_of_requests, index, status);
}

#endif
