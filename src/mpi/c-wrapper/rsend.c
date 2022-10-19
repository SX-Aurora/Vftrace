#ifdef _MPI
#include <mpi.h>

#include "mpi_logging.h"
#include "rsend_c2vftr.h"

int MPI_Rsend(const void *buf, int count, MPI_Datatype datatype,
              int dest, int tag, MPI_Comm comm) {
   if (vftr_no_mpi_logging()) {
      return PMPI_Rsend(buf, count, datatype, dest, tag, comm);
   } else {
      return vftr_MPI_Rsend_c2vftr(buf, count, datatype, dest, tag, comm);
   }
}

#endif
