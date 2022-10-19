#ifdef _MPI
#include <mpi.h>

#include "mpi_logging.h"
#include "sendrecv_replace_c2vftr.h"

int MPI_Sendrecv_replace(void *buf, int count, MPI_Datatype datatype,
                         int dest, int sendtag, int source, int recvtag,
                         MPI_Comm comm, MPI_Status *status) {
   if (vftr_no_mpi_logging()) {
      return PMPI_Sendrecv_replace(buf, count, datatype, dest, sendtag,
                                   source, recvtag, comm, status);
   } else {
      return vftr_MPI_Sendrecv_replace_c2vftr(buf, count, datatype, dest, sendtag,
                                              source, recvtag, comm, status);
   }
}

#endif
