#ifdef _MPI
#include <mpi.h>

#include "mpi_buf_addr_const.h"
#include "ialltoall.h"

int vftr_MPI_Ialltoall_c2vftr(const void *sendbuf, int sendcount,
                              MPI_Datatype sendtype, void *recvbuf,
                              int recvcount, MPI_Datatype recvtype,
                              MPI_Comm comm, MPI_Request *request) {
   // determine if inter or intra communicator
   int isintercom;
   PMPI_Comm_test_inter(comm, &isintercom);
   if (isintercom) {
      return vftr_MPI_Ialltoall_intercom(sendbuf, sendcount, sendtype,
                                         recvbuf, recvcount, recvtype,
                                         comm, request);
   } else {
      if (vftr_is_C_MPI_IN_PLACE(sendbuf)) {
         return vftr_MPI_Ialltoall_inplace(sendbuf, sendcount, sendtype,
                                           recvbuf, recvcount, recvtype,
                                           comm, request);
      } else {
         return vftr_MPI_Ialltoall(sendbuf, sendcount, sendtype,
                                   recvbuf, recvcount, recvtype,
                                   comm, request);
      }
   }
}

#endif
