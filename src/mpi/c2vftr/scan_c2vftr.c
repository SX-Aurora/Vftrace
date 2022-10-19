#ifdef _MPI
#include <mpi.h>

#include "scan.h"

int vftr_MPI_Scan_c2vftr(const void *sendbuf, void *recvbuf, int count,
                         MPI_Datatype datatype, MPI_Op op, MPI_Comm comm) {
   return vftr_MPI_Scan(sendbuf, recvbuf, count, datatype, op, comm);
}

#endif
