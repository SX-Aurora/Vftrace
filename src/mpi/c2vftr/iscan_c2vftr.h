#ifndef ISCAN_C2VFTR_H
#define ISCAN_C2VFTR_H

#ifdef _MPI
#include <mpi.h>

int vftr_MPI_Iscan_c2vftr(const void *sendbuf, void *recvbuf, int count,
                          MPI_Datatype datatype, MPI_Op op, MPI_Comm comm,
                          MPI_Request *request);

#endif
#endif
