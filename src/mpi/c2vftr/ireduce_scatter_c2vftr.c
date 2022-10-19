#ifdef _MPI
#include <stdlib.h>

#include <mpi.h>

#include "mpi_buf_addr_const.h"
#include "ireduce_scatter.h"

int vftr_MPI_Ireduce_scatter_c2vftr(const void *sendbuf, void *recvbuf,
                                    const int *recvcounts, MPI_Datatype datatype,
                                    MPI_Op op, MPI_Comm comm, MPI_Request *request) {
   // create a copy of recvcount.
   // It will be deallocated upon completion of the request
   int size;
   PMPI_Comm_size(comm, &size);
   int *tmp_recvcounts = (int*) malloc(size*sizeof(int));
   for (int i=0; i<size; i++) {
      tmp_recvcounts[i] = recvcounts[i];
   }
   // determine if inter or intra communicator
   int isintercom;
   PMPI_Comm_test_inter(comm, &isintercom);
   if (isintercom) {
      return vftr_MPI_Ireduce_scatter_intercom(sendbuf, recvbuf, tmp_recvcounts,
                                               datatype, op, comm, request);
   } else {
      if (vftr_is_C_MPI_IN_PLACE(sendbuf)) {
         return vftr_MPI_Ireduce_scatter_inplace(sendbuf, recvbuf, tmp_recvcounts,
                                                 datatype, op, comm, request);
      } else {
         return vftr_MPI_Ireduce_scatter(sendbuf, recvbuf, tmp_recvcounts,
                                         datatype, op, comm, request);
      }
   }
}

#endif
