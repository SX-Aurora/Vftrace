#ifndef IRECV_C2VFTR_H
#define IRECV_C2VFTR_H

#ifdef _MPI
#include <mpi.h>

int vftr_MPI_Irecv_c2vftr(void *buf, int count, MPI_Datatype datatype,
                          int source, int tag, MPI_Comm comm, MPI_Request *request);

#endif
#endif
