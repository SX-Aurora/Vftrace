#ifndef TEST_H
#define TEST_H

#include <mpi.h>

int vftr_MPI_Test(MPI_Request *request, int *flag, MPI_Status *status);

#endif
