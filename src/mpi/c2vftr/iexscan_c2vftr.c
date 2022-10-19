#ifdef _MPI
#include <mpi.h>

#include "iexscan.h"

int vftr_MPI_Iexscan_c2vftr(const void *sendbuf, void *recvbuf, int count,
                            MPI_Datatype datatype, MPI_Op op, MPI_Comm comm,
                            MPI_Request *request) {
   return vftr_MPI_Iexscan(sendbuf, recvbuf, count, datatype, op, comm, request);
}

#endif
