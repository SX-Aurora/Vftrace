#ifdef _MPI
#include <mpi.h>

#include "mpi_logging.h"
#include "test_c2vftr.h"

int MPI_Test(MPI_Request *request, int *flag, MPI_Status *status) {
   if (vftr_no_mpi_logging()) {
      return PMPI_Test(request, flag, status);
   } else {
      return vftr_MPI_Test_c2vftr(request, flag, status);
   }
}

#endif
