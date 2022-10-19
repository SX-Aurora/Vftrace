#ifdef _MPI
#include <mpi.h>

#include "mpi_buf_addr_const.h"
#include "allreduce.h"

int vftr_MPI_Allreduce_c2vftr(const void *sendbuf, void *recvbuf, int count,
                              MPI_Datatype datatype, MPI_Op op, MPI_Comm comm) {
   int isintercom;
   PMPI_Comm_test_inter(comm, &isintercom);
   if (isintercom) {
      return vftr_MPI_Allreduce_intercom(sendbuf, recvbuf,
                                         count, datatype, op, comm);
   } else {
      if (vftr_is_C_MPI_IN_PLACE(sendbuf)) {
         return vftr_MPI_Allreduce_inplace(sendbuf, recvbuf,
                                           count, datatype, op, comm);
      } else {
         return vftr_MPI_Allreduce(sendbuf, recvbuf,
                                   count, datatype, op, comm);
      }
   }
}

#endif
