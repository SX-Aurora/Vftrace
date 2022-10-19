#ifdef _MPI
#include <mpi.h>

#include "init_thread.h"

int vftr_MPI_Init_thread_c2vftr(int *argc, char ***argv,
                                int provided, int *required) {
   return vftr_MPI_Init_thread(argc, argv, provided, required);
}

#endif
