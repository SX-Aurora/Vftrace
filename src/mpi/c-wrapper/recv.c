#ifdef _MPI
#include <mpi.h>

#include "mpi_logging.h"
#include "recv_c2vftr.h"

int MPI_Recv(void *buf, int count, MPI_Datatype datatype,
             int source, int tag, MPI_Comm comm, MPI_Status *status) {
   if (vftr_no_mpi_logging()) {
      return PMPI_Recv(buf, count, datatype, source, tag, comm, status);
   } else {
      return vftr_MPI_Recv_c2vftr(buf, count, datatype, source, tag, comm, status);
   }
}

#endif
