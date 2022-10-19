#ifdef _MPI
#include <mpi.h>

#include "mpi_logging.h"
#include "sync_time.h"
#include "bcast_c2vftr.h"

int MPI_Bcast(void *buffer, int count, MPI_Datatype datatype,
              int root, MPI_Comm comm) {

   vftr_estimate_sync_time("MPI_Bcast_sync", comm);

   if (vftr_no_mpi_logging()) {
      return PMPI_Bcast(buffer, count, datatype, root, comm);
   } else {
      return vftr_MPI_Bcast_c2vftr(buffer, count, datatype, root, comm);
   }
}

#endif
