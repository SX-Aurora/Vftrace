#ifdef _MPI
#include <mpi.h>

#include "pcontrol.h"

int vftr_MPI_Pcontrol_c2vftr(const int level, ...) {
   return vftr_MPI_Pcontrol(level);
}

#endif
