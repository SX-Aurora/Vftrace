#ifdef _MPI
#include <mpi.h>

#include "mpi_logging.h"
#include "irecv_c2vftr.h"

int MPI_Irecv(void *buf, int count, MPI_Datatype datatype,
              int source, int tag, MPI_Comm comm, MPI_Request *request) {
   if (vftr_no_mpi_logging()) {
      return PMPI_Irecv(buf, count, datatype, source, tag, comm, request);
   } else {
      return vftr_MPI_Irecv_c2vftr(buf, count, datatype, source, tag, comm, request);
   }
}

#endif
