#ifdef _MPI
#include <mpi.h>

#include "init_thread_c2vftr.h"

int MPI_Init_thread(int *argc, char ***argv,
                    int required, int *provided) {
   return vftr_MPI_Init_c2vftr(argc, argv, required, provided);
}

#endif
