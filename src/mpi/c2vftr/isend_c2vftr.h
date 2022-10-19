#ifndef ISEND_C2VFTR_H
#define ISEND_C2VFTR_H

#ifdef _MPI
#include <mpi.h>

int vftr_MPI_Isend_c2vftr(const void *buf, int count, MPI_Datatype datatype,
                          int dest, int tag, MPI_Comm comm,
                          MPI_Request *request);

#endif
#endif
