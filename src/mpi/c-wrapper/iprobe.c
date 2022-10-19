#ifdef _MPI
#include <mpi.h>

#include "mpi_logging.h"
#include "iprobe_c2vftr.h"

int MPI_Iprobe(int source, int tag, MPI_Comm comm,
               int *flag, MPI_Status *status) {
   if (vftr_no_mpi_logging()) {
      return PMPI_Iprobe(source, tag, comm, flag, status);
   } else {
      return vftr_MPI_Iprobe_c2vftr(source, tag, comm, flag, status);
   }
}

#endif
