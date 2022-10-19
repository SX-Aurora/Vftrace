#ifndef START_C2VFTR_H
#define START_C2VFTR_H

#ifdef _MPI
#include <mpi.h>

int vftr_MPI_Start_c2vftr(MPI_Request *request);

#endif
#endif
