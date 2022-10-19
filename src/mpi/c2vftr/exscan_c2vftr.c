#ifdef _MPI
#include <mpi.h>

#include "exscan.h"

int vftr_MPI_Exscan_c2vftr(const void *sendbuf, void *recvbuf, int count,
                           MPI_Datatype datatype, MPI_Op op, MPI_Comm comm) {
   return vftr_MPI_Exscan(sendbuf, recvbuf, count, datatype, op, comm);
}

#endif
