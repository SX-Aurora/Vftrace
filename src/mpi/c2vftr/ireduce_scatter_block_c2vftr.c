#ifdef _MPI
#include <mpi.h>

#include "mpi_buf_addr_const.h"
#include "ireduce_scatter_block.h"

int vftr_MPI_Ireduce_scatter_block_c2vftr(const void *sendbuf, void *recvbuf,
                                          int recvcount, MPI_Datatype datatype,
                                          MPI_Op op, MPI_Comm comm,
                                          MPI_Request *request) {
   // determine if inter or intra communicator
   int isintercom;
   PMPI_Comm_test_inter(comm, &isintercom);
   if (isintercom) {
      return vftr_MPI_Ireduce_scatter_block_intercom(sendbuf, recvbuf,
                                                     recvcount, datatype,
                                                     op, comm, request);
   } else {
      if (vftr_is_C_MPI_IN_PLACE(sendbuf)) {
         return vftr_MPI_Ireduce_scatter_block_inplace(sendbuf, recvbuf,
                                                       recvcount, datatype,
                                                       op, comm, request);
      } else {
         return vftr_MPI_Ireduce_scatter_block(sendbuf, recvbuf,
                                               recvcount, datatype,
                                               op, comm, request);
      }
   }
}

#endif
