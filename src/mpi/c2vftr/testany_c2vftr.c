#ifdef _MPI
#include <mpi.h>

#include "testany.h"

int vftr_MPI_Testany_c2vftr(int count, MPI_Request array_of_requests[],
                            int *index, int *flag, MPI_Status *status) {
   return vftr_MPI_Testany(count, array_of_requests, index, flag, status);
}

#endif
