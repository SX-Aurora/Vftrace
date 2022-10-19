#ifdef _MPI
#include <mpi.h>

#include "mpi_logging.h"
#include "recv_init_c2vftr.h"

int MPI_Recv_init(void *buf, int count, MPI_Datatype datatype,
                  int source, int tag, MPI_Comm comm,
                  MPI_Request *request) {
   if (vftr_no_mpi_logging()) {
      return PMPI_Recv_init(buf, count, datatype, source, tag, comm, request);
   } else {
      return vftr_MPI_Recv_init_c2vftr(buf, count, datatype, source, tag, comm, request);
   }
}

#endif
