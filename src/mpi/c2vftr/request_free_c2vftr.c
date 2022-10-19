#ifdef _MPI
#include <mpi.h>

#include "request_free.h"

int vftr_MPI_Request_free_c2vftr(MPI_Request *request) {
   return vftr_MPI_Request_free(request);
}

#endif
