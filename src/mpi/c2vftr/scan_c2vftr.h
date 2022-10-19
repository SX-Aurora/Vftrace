#ifndef SCAN_C2VFTR_H
#define SCAN_C2VFTR_H

#ifdef _MPI
#include <mpi.h>

int vftr_MPI_Scan_c2vftr(const void *sendbuf, void *recvbuf, int count,
                         MPI_Datatype datatype, MPI_Op op, MPI_Comm comm);

#endif
#endif
