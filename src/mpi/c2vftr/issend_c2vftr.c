#ifdef _MPI
#include <mpi.h>

#include "issend.h"

int vftr_MPI_Issend_c2vftr(const void *buf, int count, MPI_Datatype datatype,
                           int dest, int tag, MPI_Comm comm,
                           MPI_Request *request) {
   return vftr_MPI_Issend(buf, count, datatype, dest, tag, comm, request);
}

#endif
