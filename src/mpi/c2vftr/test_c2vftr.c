#ifdef _MPI
#include <mpi.h>

#include "test.h"

int vftr_MPI_Test_c2vftr(MPI_Request *request, int *flag, MPI_Status *status) {
   return vftr_MPI_Test(request, flag, status);
}

#endif
