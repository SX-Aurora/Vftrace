#ifdef _MPI
#include <mpi.h>

#include "barrier.h"

int vftr_MPI_Barrier_c2vftr(MPI_Comm comm) {
   return vftr_MPI_Barrier(comm);
}

#endif
