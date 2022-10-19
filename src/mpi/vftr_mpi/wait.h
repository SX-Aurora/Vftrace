#ifndef WAIT_H
#define WAIT_H

#include <mpi.h>

int vftr_MPI_Wait(MPI_Request *request, MPI_Status *status);

#endif
