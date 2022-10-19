#ifdef _MPI
#include <mpi.h>

#include "mpi_logging.h"
#include "bsend_c2vftr.h"

int MPI_Bsend(const void *buf, int count, MPI_Datatype datatype,
              int dest, int tag, MPI_Comm comm) {
   if (vftr_no_mpi_logging()) {
      return PMPI_Bsend(buf, count, datatype, dest, tag, comm);
   } else {
      return vftr_MPI_Bsend_c2vftr(buf, count, datatype, dest, tag, comm);
   }
}

#endif
