#ifndef SCAN_H
#define SCAN_H

#include <mpi.h>

int vftr_MPI_Scan(const void *sendbuf, void *recvbuf, int count,
                  MPI_Datatype datatype, MPI_Op op, MPI_Comm comm);

#endif
