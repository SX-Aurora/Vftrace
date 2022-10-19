#ifdef _MPI
#include <mpi.h>

#include "startall.h"

int vftr_MPI_Startall_c2vftr(int count, MPI_Request *array_of_requests) {
   return vftr_MPI_Startall(count, array_of_requests);
}

#endif
