#ifdef _MPI
#include <mpi.h>

#include "recv_init.h"

int vftr_MPI_Recv_init_c2vftr(void *buf, int count, MPI_Datatype datatype,
                              int source, int tag, MPI_Comm comm,
                              MPI_Request *request) {
   return vftr_MPI_Recv_init(buf, count, datatype, source, tag, comm, request);
}

#endif
