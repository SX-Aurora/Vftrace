#ifndef STARTALL_H
#define STARTALL_H

#include <mpi.h>

int vftr_MPI_Startall(int count, MPI_Request *array_of_requests);

#endif
