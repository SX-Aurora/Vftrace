#ifdef _MPI
#include <mpi.h>

#include "irsend.h"

int vftr_MPI_Irsend_c2vftr(const void *buf, int count, MPI_Datatype datatype,
                           int dest, int tag, MPI_Comm comm,
                           MPI_Request *request) {
   return vftr_MPI_Irsend(buf, count, datatype, dest, tag, comm, request);
}

#endif
