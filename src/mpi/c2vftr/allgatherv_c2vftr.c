#ifdef _MPI
#include <mpi.h>

#include "mpi_buf_addr_const.h"
#include "allgatherv.h"

int vftr_MPI_Allgatherv_c2vftr(const void *sendbuf, int sendcount,
                               MPI_Datatype sendtype, void *recvbuf,
                               const int *recvcounts, const int *displs,
                               MPI_Datatype recvtype, MPI_Comm comm) {
   // determine if inter or intra communicator
   int isintercom;
   PMPI_Comm_test_inter(comm, &isintercom);
   if (isintercom) {
      return vftr_MPI_Allgatherv_intercom(sendbuf, sendcount, sendtype,
                                          recvbuf, recvcounts, displs,
                                          recvtype, comm);
   } else {
      if (vftr_is_C_MPI_IN_PLACE(sendbuf)) {
         return vftr_MPI_Allgatherv_inplace(sendbuf, sendcount, sendtype,
                                            recvbuf, recvcounts, displs,
                                            recvtype, comm);
      } else {
         return vftr_MPI_Allgatherv(sendbuf, sendcount, sendtype,
                                    recvbuf, recvcounts, displs,
                                    recvtype, comm);
      }
   }
}

#endif
