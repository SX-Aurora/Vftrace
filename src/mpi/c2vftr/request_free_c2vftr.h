#ifndef REQUEST_FREE_C2VFTR_H
#define REQUEST_FREE_C2VFTR_H

#ifdef _MPI
#include <mpi.h>

int vftr_MPI_Request_free_c2vftr(MPI_Request *request);

#endif
#endif
