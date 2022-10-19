#ifdef _MPI
#include <mpi.h>

#include "recv.h"

int vftr_MPI_Recv_c2vftr(void *buf, int count, MPI_Datatype datatype,
                         int source, int tag, MPI_Comm comm, MPI_Status *status) {
   return vftr_MPI_Recv(buf, count, datatype, source, tag, comm, status);
}

#endif
