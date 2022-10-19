#ifdef _MPI
#include <mpi.h>

#include "wait.h"

int vftr_MPI_Wait_c2vftr(MPI_Request *request, MPI_Status *status) {
   return vftr_MPI_Wait(request, status);
}

#endif
