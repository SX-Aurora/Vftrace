#ifndef BARRIER_C2VFTR_H
#define BARRIER_C2VFTR_H

#ifdef _MPI
#include <mpi.h>

int vftr_MPI_Barrier_c2vftr(MPI_Comm comm);

#endif
#endif
