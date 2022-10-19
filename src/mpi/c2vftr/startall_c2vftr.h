#ifndef STARTALL_C2VFTR_H
#define STARTALL_C2VFTR_H

#ifdef _MPI
#include <mpi.h>

int vftr_MPI_Startall_c2vftr(int count, MPI_Request *array_of_requests);

#endif
#endif
