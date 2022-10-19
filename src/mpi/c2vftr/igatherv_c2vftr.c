#ifdef _MPI
#include <stdlib.h>

#include <mpi.h>

#include "mpi_buf_addr_const.h"
#include "igatherv.h"

int vftr_MPI_Igatherv_c2vftr(const void *sendbuf, int sendcount,
                             MPI_Datatype sendtype, void *recvbuf,
                             const int *recvcounts, const int *displs,
                             MPI_Datatype recvtype, int root,
                             MPI_Comm comm, MPI_Request *request) {
   // create a copy of recvcounts and displacements
   // They will be deallocated upon completion of the request
   int isroot;
   int size;
   int isintercom;
   PMPI_Comm_test_inter(comm, &isintercom);
   if (isintercom) {
      isroot = MPI_ROOT == root;
   } else {
      int myrank;
      PMPI_Comm_rank(comm, &myrank);
      isroot = myrank == root;
   }

   int *tmp_recvcounts = NULL ;
   int *tmp_displs = NULL ;
   if (isroot) {
      if (isintercom) {
         PMPI_Comm_remote_size(comm, &size);
      } else {
         PMPI_Comm_size(comm, &size);
      }
      tmp_recvcounts = (int*) malloc(size*sizeof(int));
      for (int i=0; i<size; i++) {
        tmp_recvcounts[i] = recvcounts[i];
      }
      tmp_displs = (int*) malloc(size*sizeof(int));
      for (int i=0; i<size; i++) {
         tmp_displs[i] = displs[i];
      }
   }

   if (isintercom) {
      return vftr_MPI_Igatherv_intercom(sendbuf, sendcount, sendtype,
                                        recvbuf, tmp_recvcounts, tmp_displs,
                                        recvtype, root, comm, request);
   } else {
      if (vftr_is_C_MPI_IN_PLACE(sendbuf)) {
         return vftr_MPI_Igatherv_inplace(sendbuf, sendcount, sendtype,
                                          recvbuf, tmp_recvcounts, tmp_displs,
                                          recvtype, root, comm, request);
      } else {
         return vftr_MPI_Igatherv(sendbuf, sendcount, sendtype,
                                  recvbuf, tmp_recvcounts, tmp_displs,
                                  recvtype, root, comm, request);
      }
   }
}

#endif
