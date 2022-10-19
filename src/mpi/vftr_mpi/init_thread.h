#ifndef INIT_THREAD_H
#define INIT_THREAD_H

#include <mpi.h>

int vftr_MPI_Init_thread(int *argc, char ***argv,
                         int required, int *provided);

#endif
