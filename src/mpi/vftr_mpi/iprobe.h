#ifndef IPROBE_H
#define IPROBE_H

#include <mpi.h>

int vftr_MPI_Iprobe(int source, int tag, MPI_Comm comm, int *flag, MPI_Status *status);

#endif
