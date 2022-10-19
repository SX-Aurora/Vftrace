#ifdef _MPI
#include <mpi.h>

#include "init.h"

int vftr_MPI_Init_c2vftr(int *argc, char ***argv) {
   return vftr_MPI_Init(argc, argv);
}

#endif
