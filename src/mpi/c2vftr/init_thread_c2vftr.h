#ifndef INIT_THREAD_C2VFTR_H
#define INIT_THREAD_C2VFTR_H

#ifdef _MPI
#include <mpi.h>

int vftr_MPI_Init_c2vftr(int *argc, char ***argv,
                         int required, int *provided);

#endif
#endif
