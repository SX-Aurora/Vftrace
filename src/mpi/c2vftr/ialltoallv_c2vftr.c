#ifdef _MPI
#include <stdlib.h>

#include <mpi.h>

#include "mpi_buf_addr_const.h"
#include "ialltoallv.h"

int vftr_MPI_Ialltoallv_c2vftr(const void *sendbuf, const int *sendcounts,
                               const int *sdispls, MPI_Datatype sendtype,
                               void *recvbuf, const int *recvcounts,
                               const int *rdispls, MPI_Datatype recvtype,
                               MPI_Comm comm, MPI_Request *request) {
   int size;
   int isintercom;
   PMPI_Comm_test_inter(comm, &isintercom);
   if (isintercom) {
      PMPI_Comm_remote_size(comm, &size);
   } else {
      PMPI_Comm_size(comm, &size);
   }

   int *tmp_sendcounts = NULL;
   int *tmp_sdispls = NULL;
   // sendcounts and senddisplacements are ignored
   // if sendbuffer is MPI_IN_PLACE
   if (!vftr_is_C_MPI_IN_PLACE(sendbuf)) {
      tmp_sendcounts = (int*) malloc(size*sizeof(int));
      for (int i=0; i<size; i++) {
         tmp_sendcounts[i] = sendcounts[i];
      }
      tmp_sdispls = (int*) malloc(size*sizeof(int));
      for (int i=0; i<size; i++) {
         tmp_sdispls[i] = sdispls[i];
      }
   }
   int *tmp_recvcounts = (int*) malloc(size*sizeof(int));
   for (int i=0; i<size; i++) {
      tmp_recvcounts[i] = recvcounts[i];
   }
   int *tmp_rdispls = (int*) malloc(size*sizeof(int));
   for (int i=0; i<size; i++) {
      tmp_rdispls[i] = rdispls[i];
   }

   if (isintercom) {
      return vftr_MPI_Ialltoallv_intercom(sendbuf, tmp_sendcounts,
                                          tmp_sdispls, sendtype,
                                          recvbuf, tmp_recvcounts,
                                          tmp_rdispls, recvtype,
                                          comm, request);
   } else {
      if (vftr_is_C_MPI_IN_PLACE(sendbuf)) {
         return vftr_MPI_Ialltoallv_inplace(sendbuf, tmp_sendcounts,
                                            tmp_sdispls, sendtype,
                                            recvbuf, tmp_recvcounts,
                                            tmp_rdispls, recvtype,
                                            comm, request);
      } else {
         return vftr_MPI_Ialltoallv(sendbuf, tmp_sendcounts,
                                    tmp_sdispls, sendtype,
                                    recvbuf, tmp_recvcounts,
                                    tmp_rdispls, recvtype,
                                    comm, request);
      }
   }
}

#endif
