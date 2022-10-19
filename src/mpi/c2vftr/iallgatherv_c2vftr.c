#ifdef _MPI
#include <stdlib.h>

#include <mpi.h>

#include "mpi_buf_addr_const.h"
#include "iallgatherv.h"

int vftr_MPI_Iallgatherv_c2vftr(const void *sendbuf, int sendcount,
                                MPI_Datatype sendtype, void *recvbuf,
                                const int *recvcounts, const int *displs,
                                MPI_Datatype recvtype, MPI_Comm comm,
                                MPI_Request *request) {
   // create a copy of recvcount.
   // It will be deallocated upon completion of the request
   int size;
   int isintercom;
   PMPI_Comm_test_inter(comm, &isintercom);
   if (isintercom) {
      PMPI_Comm_remote_size(comm, &size);
   } else {
      PMPI_Comm_size(comm, &size);
   }
   int *tmp_recvcounts = (int*) malloc(size*sizeof(int));
   for (int i=0; i<size; i++) {
      tmp_recvcounts[i] = recvcounts[i];
   }
   int *tmp_displs = (int*) malloc(size*sizeof(int));
   for (int i=0; i<size; i++) {
      tmp_displs[i] = displs[i];
   }

   // determine if inter or intra communicator
   if (isintercom) {
      return vftr_MPI_Iallgatherv_intercom(sendbuf, sendcount, sendtype,
                                           recvbuf, tmp_recvcounts, tmp_displs,
                                           recvtype, comm, request);
   } else {
      if (vftr_is_C_MPI_IN_PLACE(sendbuf)) {
         return vftr_MPI_Iallgatherv_inplace(sendbuf, sendcount, sendtype,
                                             recvbuf, tmp_recvcounts, tmp_displs,
                                             recvtype, comm, request);
      } else {
         return vftr_MPI_Iallgatherv(sendbuf, sendcount, sendtype,
                                     recvbuf, tmp_recvcounts, tmp_displs,
                                     recvtype, comm, request);
      }
   }
}

#endif
