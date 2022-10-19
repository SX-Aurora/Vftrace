#ifdef _MPI
#include <mpi.h>

#include "mpi_logging.h"
#include "ssend_init_c2vftr.h"

int MPI_Ssend_init(const void *buf, int count, MPI_Datatype datatype,
                   int dest, int tag, MPI_Comm comm,
                   MPI_Request *request) {
   if (vftr_no_mpi_logging()) {
      return PMPI_Ssend_init(buf, count, datatype, dest, tag, comm, request);
   } else {
      return vftr_MPI_Ssend_init_c2vftr(buf, count, datatype, dest, tag, comm, request);
   }
}

#endif
