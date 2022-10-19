#ifdef _MPI
#include <mpi.h>

#include "start.h"

int vftr_MPI_Start_c2vftr(MPI_Request *request) {
      return vftr_MPI_Start(request);
}

#endif
