#ifdef _MPI
#include <mpi.h>

#include "mpi_buf_addr_const.h"
#include "alltoallv.h"

int vftr_MPI_Alltoallv_c2vftr(const void *sendbuf, const int *sendcounts,
                              const int *sdispls, MPI_Datatype sendtype,
                              void *recvbuf, const int *recvcounts,
                              const int *rdispls, MPI_Datatype recvtype,
                              MPI_Comm comm) {
   // determine if inter or intra communicator
   int isintercom;
   PMPI_Comm_test_inter(comm, &isintercom);
   if (isintercom) {
      return vftr_MPI_Alltoallv_intercom(sendbuf, sendcounts,
                                         sdispls, sendtype,
                                         recvbuf, recvcounts,
                                         rdispls, recvtype,
                                         comm);
   } else {
      if (vftr_is_C_MPI_IN_PLACE(sendbuf)) {
         return vftr_MPI_Alltoallv_inplace(sendbuf, sendcounts,
                                           sdispls, sendtype,
                                           recvbuf, recvcounts,
                                           rdispls, recvtype,
                                           comm);
      } else {
         return vftr_MPI_Alltoallv(sendbuf, sendcounts,
                                   sdispls, sendtype,
                                   recvbuf, recvcounts,
                                   rdispls, recvtype,
                                   comm);
      }
   }
}

#endif
