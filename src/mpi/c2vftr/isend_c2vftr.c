#ifdef _MPI
#include <mpi.h>

#include "isend.h"

int vftr_MPI_Isend_c2vftr(const void *buf, int count, MPI_Datatype datatype,
                          int dest, int tag, MPI_Comm comm,
                          MPI_Request *request) {
   return vftr_MPI_Isend(buf, count, datatype, dest, tag, comm, request);
}

#endif
