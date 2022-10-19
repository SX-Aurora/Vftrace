#ifdef _MPI
#include <mpi.h>

#include "mpi_logging.h"
#include "probe_c2vftr.h"

int MPI_Probe(int source, int tag, MPI_Comm comm, MPI_Status *status) {
   if (vftr_no_mpi_logging()) {
      return PMPI_Probe(source, tag, comm, status);
   } else {
      return vftr_MPI_Probe_c2vftr(source, tag, comm, status);
   }
}

#endif
