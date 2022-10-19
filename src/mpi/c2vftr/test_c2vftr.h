#ifndef TEST_C2VFTR_H
#define TEST_C2VFTR_H

#ifdef _MPI
#include <mpi.h>

int vftr_MPI_Test_c2vftr(MPI_Request *request, int *flag, MPI_Status *status);

#endif
#endif
