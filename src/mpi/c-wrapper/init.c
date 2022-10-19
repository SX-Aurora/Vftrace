#ifdef _MPI
#include <mpi.h>

#include "init_c2vftr.h"

int MPI_Init(int *argc, char ***argv) {
   return vftr_MPI_Init_c2vftr(argc, argv);
}

#endif
