#ifdef _MPI
#include <mpi.h>

#include <stdlib.h>

#include "mpi_buf_addr_const.h"
#include "ialltoallw.h"

void vftr_MPI_Ialltoallw_f2vftr(void *sendbuf, MPI_Fint *f_sendcounts,
                                MPI_Fint *f_sdispls, MPI_Fint *f_sendtypes,
                                void *recvbuf, MPI_Fint *f_recvcounts,
                                MPI_Fint *f_rdispls, MPI_Fint *f_recvtypes,
                                MPI_Fint *f_comm, MPI_Fint *f_request,
                                MPI_Fint *f_error) {

   MPI_Comm c_comm = PMPI_Comm_f2c(*f_comm);

   int size;
   int isintercom;
   PMPI_Comm_test_inter(c_comm, &isintercom);
   if (isintercom) {
      PMPI_Comm_remote_size(c_comm, &size);
   } else {
      PMPI_Comm_size(c_comm, &size);
   }

   int *c_sendcounts = (int*) malloc(size*sizeof(int));
   for (int i=0; i<size; i++) {
      c_sendcounts[i] = (int) f_sendcounts[i];
   }
   int *c_sdispls = (int*) malloc(size*sizeof(int));
   for (int i=0; i<size; i++) {
      c_sdispls[i] = (int) f_sdispls[i];
   }
   MPI_Datatype *c_sendtypes = (MPI_Datatype*) malloc(size*sizeof(MPI_Datatype));
   for (int i=0; i<size; i++) {
      c_sendtypes[i] = PMPI_Type_f2c(f_sendtypes[i]);
   }

   int *c_recvcounts = (int*) malloc(size*sizeof(int));
   for (int i=0; i<size; i++) {
      c_recvcounts[i] = (int) f_recvcounts[i];
   }
   int *c_rdispls = (int*) malloc(size*sizeof(int));
   for (int i=0; i<size; i++) {
      c_rdispls[i] = (int) f_rdispls[i];
   }
   MPI_Datatype *c_recvtypes = (MPI_Datatype*) malloc(size*sizeof(MPI_Datatype));
   for (int i=0; i<size; i++) {
      c_recvtypes[i] = PMPI_Type_f2c(f_recvtypes[i]);
   }

   MPI_Request c_request;

   sendbuf = (void*) vftr_is_F_MPI_IN_PLACE(sendbuf) ? MPI_IN_PLACE : sendbuf;
   sendbuf = (void*) vftr_is_F_MPI_BOTTOM(sendbuf) ? MPI_BOTTOM : sendbuf;
   recvbuf = (void*) vftr_is_F_MPI_BOTTOM(recvbuf) ? MPI_BOTTOM : recvbuf;

   int c_error;
   if (isintercom) {
      c_error = vftr_MPI_Ialltoallw_intercom(sendbuf,
                                             c_sendcounts,
                                             c_sdispls,
                                             c_sendtypes,
                                             recvbuf,
                                             c_recvcounts,
                                             c_rdispls,
                                             c_recvtypes,
                                             c_comm,
                                             &c_request);
   } else {
      if (vftr_is_C_MPI_IN_PLACE(sendbuf)) {
         c_error = vftr_MPI_Ialltoallw_inplace(sendbuf,
                                               c_sendcounts,
                                               c_sdispls,
                                               c_sendtypes,
                                               recvbuf,
                                               c_recvcounts,
                                               c_rdispls,
                                               c_recvtypes,
                                               c_comm,
                                               &c_request);
      } else {
         c_error = vftr_MPI_Ialltoallw(sendbuf,
                                       c_sendcounts,
                                       c_sdispls,
                                       c_sendtypes,
                                       recvbuf,
                                       c_recvcounts,
                                       c_rdispls,
                                       c_recvtypes,
                                       c_comm,
                                       &c_request);
      }
   }

   *f_error = (MPI_Fint) (c_error);
   *f_request = PMPI_Request_c2f(c_request);
}

#endif
