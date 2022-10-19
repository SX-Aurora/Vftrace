#ifndef WAIT_C2VFTR_H
#define WAIT_C2VFTR_H

#ifdef _MPI
#include <mpi.h>

int vftr_MPI_Wait_c2vftr(MPI_Request *request, MPI_Status *status);

#endif
#endif
