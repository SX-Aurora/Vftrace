#ifdef _MPI
#include <mpi.h>

#include "mpi_buf_addr_const.h"
#include "scatter.h"

int vftr_MPI_Scatter_c2vftr(const void *sendbuf, int sendcount,
                            MPI_Datatype sendtype, void *recvbuf,
                            int recvcount, MPI_Datatype recvtype,
                            int root, MPI_Comm comm) {
   // determine if inter or intra communicator
   int isintercom;
   PMPI_Comm_test_inter(comm, &isintercom);
   if (isintercom) {
      return vftr_MPI_Scatter_intercom(sendbuf, sendcount, sendtype,
                                       recvbuf, recvcount, recvtype,
                                       root, comm);
   } else {
      if (vftr_is_C_MPI_IN_PLACE(recvbuf)) {
         return vftr_MPI_Scatter_inplace(sendbuf, sendcount, sendtype,
                                         recvbuf, recvcount, recvtype,
                                         root, comm);
      } else {
         return vftr_MPI_Scatter(sendbuf, sendcount, sendtype,
                                 recvbuf, recvcount, recvtype,
                                 root, comm);
      }
   }
}

#endif
