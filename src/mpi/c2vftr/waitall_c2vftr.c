#ifdef _MPI
#include <mpi.h>

#include "waitall.h"

int vftr_MPI_Waitall_c2vftr(int count, MPI_Request array_of_requests[],
                            MPI_Status array_of_statuses[]) {
   return vftr_MPI_Waitall(count, array_of_requests, array_of_statuses);
}

#endif
