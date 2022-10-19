#ifdef _MPI
#include <mpi.h>

#include "iscan.h"

int vftr_MPI_Iscan_c2vftr(const void *sendbuf, void *recvbuf, int count,
                          MPI_Datatype datatype, MPI_Op op, MPI_Comm comm,
                          MPI_Request *request) {
   return vftr_MPI_Iscan(sendbuf, recvbuf, count, datatype, op, comm, request);
}

#endif
