#ifdef _MPI
#include <mpi.h>

#include "pcontrol_c2vftr.h"

int MPI_Pcontrol(const int level, ...) {
   return vftr_MPI_Pcontrol_c2vftr(level);
}

#endif
