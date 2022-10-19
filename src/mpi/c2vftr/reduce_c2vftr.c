#ifdef _MPI
#include <mpi.h>

#include "mpi_buf_addr_const.h"
#include "reduce.h"

int vftr_MPI_Reduce_c2vftr(const void *sendbuf, void *recvbuf, int count,
                           MPI_Datatype datatype, MPI_Op op, int root,
                           MPI_Comm comm) {
   // test if intercommunicator
   int isintercom;
   PMPI_Comm_test_inter(comm, &isintercom);
   if (isintercom) {
      return vftr_MPI_Reduce_intercom(sendbuf, recvbuf, count,
                                      datatype, op, root, comm);
   } else {
      // if sendbuf is special address MPI_IN_PLACE
      if (vftr_is_C_MPI_IN_PLACE(sendbuf)) {
         return vftr_MPI_Reduce_inplace(sendbuf, recvbuf, count,
                                        datatype, op, root, comm);
      } else {
         return vftr_MPI_Reduce(sendbuf, recvbuf, count,
                                datatype, op, root, comm);
      }
   }
}

#endif
