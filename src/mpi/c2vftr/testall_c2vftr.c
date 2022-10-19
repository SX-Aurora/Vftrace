#ifdef _MPI
#include <mpi.h>

#include "testall.h"

int vftr_MPI_Testall_c2vftr(int count, MPI_Request array_of_requests[],
                            int *flag, MPI_Status array_of_statuses[]) {
   return vftr_MPI_Testall(count, array_of_requests, flag, array_of_statuses);
}

#endif
