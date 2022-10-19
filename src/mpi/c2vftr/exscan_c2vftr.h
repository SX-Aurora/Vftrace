#ifndef EXSCAN_C2VFTR_H
#define EXSCAN_C2VFTR_H

#ifdef _MPI
#include <mpi.h>

int vftr_MPI_Exscan_c2vftr(const void *sendbuf, void *recvbuf, int count,
                           MPI_Datatype datatype, MPI_Op op, MPI_Comm comm);

#endif
#endif
