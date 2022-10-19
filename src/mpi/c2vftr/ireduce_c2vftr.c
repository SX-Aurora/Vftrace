#ifdef _MPI
#include <mpi.h>

#include "mpi_buf_addr_const.h"
#include "ireduce.h"

int vftr_MPI_Ireduce_c2vftr(const void *sendbuf, void *recvbuf, int count,
                            MPI_Datatype datatype, MPI_Op op, int root,
                            MPI_Comm comm, MPI_Request *request) {
   // test if intercommunicator
   int isintercom;
   PMPI_Comm_test_inter(comm, &isintercom);
   if (isintercom) {
      return vftr_MPI_Ireduce_intercom(sendbuf, recvbuf, count, datatype,
                                       op, root, comm, request);
   } else {
      // if sendbuf is special address MPI_IN_PLACE
      if (vftr_is_C_MPI_IN_PLACE(sendbuf)) {
         return vftr_MPI_Ireduce_inplace(sendbuf, recvbuf, count, datatype,
                                         op, root, comm, request);
      } else {
         return vftr_MPI_Ireduce(sendbuf, recvbuf, count, datatype,
                                 op, root, comm, request);
      }
   }
}

#endif
