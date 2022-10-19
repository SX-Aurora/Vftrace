#ifndef IPROBE_C2VFTR_H
#define IPROBE_C2VFTR_H

#ifdef _MPI
#include <mpi.h>

int vftr_MPI_Iprobe_c2vftr(int source, int tag, MPI_Comm comm,
                           int *flag, MPI_Status *status);

#endif
#endif
