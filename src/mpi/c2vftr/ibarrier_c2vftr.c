#ifdef _MPI
#include <mpi.h>

#include "ibarrier.h"

int vftr_MPI_Ibarrier_c2vftr(MPI_Comm comm, MPI_Request *request) {
   return vftr_MPI_Ibarrier(comm, request);
}

#endif
