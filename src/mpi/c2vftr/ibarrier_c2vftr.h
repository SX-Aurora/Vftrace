#ifndef IBARRIER_C2VFTR_H
#define IBARRIER_C2VFTR_H

#ifdef _MPI
#include <mpi.h>

int vftr_MPI_Ibarrier_c2vftr(MPI_Comm comm, MPI_Request *request);

#endif
#endif
