#ifdef _MPI
#include <mpi.h>

#include "finalize_c2vftr.h"

int MPI_Finalize() {
   return vftr_MPI_Finalize_c2vftr();
}

#endif
