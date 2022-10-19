#ifdef _MPI
#include <mpi.h>

#include "mpi_logging.h"
#include "send_init_c2vftr.h"

int MPI_Send_init(const void *buf, int count, MPI_Datatype datatype,
                  int dest, int tag, MPI_Comm comm,
                  MPI_Request *request) {
   if (vftr_no_mpi_logging()) {
      return PMPI_Send_init(buf, count, datatype, dest, tag, comm, request);
   } else {
      return vftr_MPI_Send_init_c2vftr(buf, count, datatype, dest, tag, comm, request);
   }
}

#endif
