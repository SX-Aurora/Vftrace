#ifdef _MPI
#include <mpi.h>

#include "send_init.h"

int vftr_MPI_Send_init_c2vftr(const void *buf, int count, MPI_Datatype datatype,
                              int dest, int tag, MPI_Comm comm,
                              MPI_Request *request) {
   return vftr_MPI_Send_init(buf, count, datatype, dest, tag, comm, request);
}

#endif
