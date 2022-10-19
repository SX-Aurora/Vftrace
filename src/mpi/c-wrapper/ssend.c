#ifdef _MPI
#include <mpi.h>

#include "mpi_logging.h"
#include "ssend_c2vftr.h"

int MPI_Ssend(const void *buf, int count, MPI_Datatype datatype,
              int dest, int tag, MPI_Comm comm) {
   if (vftr_no_mpi_logging()) {
      return PMPI_Ssend(buf, count, datatype, dest, tag, comm);
   } else {
      return vftr_MPI_Ssend_c2vftr(buf, count, datatype, dest, tag, comm);
   }
}

#endif
