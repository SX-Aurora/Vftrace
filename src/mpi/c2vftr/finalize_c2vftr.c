#ifdef _MPI
#include <mpi.h>

#include "finalize.h"

int vftr_MPI_Finalize_c2vftr() {
   return vftr_MPI_Finalize();
}

#endif
