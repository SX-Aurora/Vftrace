#ifdef _MPI
#include <mpi.h>

#include <stdlib.h>

#include "mpi_buf_addr_const.h"
#include "iscatterv.h"

void vftr_MPI_Iscatterv_f082vftr(void *sendbuf, MPI_Fint *f_sendcounts, MPI_Fint *f_displs,
                                 MPI_Fint *f_sendtype, void *recvbuf, MPI_Fint *recvcount,
                                 MPI_Fint *f_recvtype, MPI_Fint *root, MPI_Fint *f_comm,
                                 MPI_Fint *f_request, MPI_Fint *f_error) {


   MPI_Comm c_comm = PMPI_Comm_f2c(*f_comm);

   // determine if root process
   int isroot;
   int size;
   int isintercom;
   PMPI_Comm_test_inter(c_comm, &isintercom);
   if (isintercom) {
      isroot = MPI_ROOT == (int) (*root);
   } else {
      int myrank;
      PMPI_Comm_rank(c_comm, &myrank);
      isroot = myrank == (int) (*root);
   }

   int *c_sendcounts = NULL;
   int *c_displs = NULL;
   if (isroot) {
      if (isintercom) {
         PMPI_Comm_remote_size(c_comm, &size);
      } else {
         PMPI_Comm_size(c_comm, &size);
      }
      c_sendcounts = (int*) malloc(size*sizeof(int));
      for (int i=0; i<size; i++) {
         c_sendcounts[i] = (int) f_sendcounts[i];
      }
      c_displs = (int*) malloc(size*sizeof(int));
      for (int i=0; i<size; i++) {
         c_displs[i] = (int) f_displs[i];
      }
   }
   MPI_Datatype c_sendtype = PMPI_Type_f2c(*f_sendtype);
   MPI_Datatype c_recvtype = PMPI_Type_f2c(*f_recvtype);
   MPI_Request c_request;

   sendbuf = (void*) vftr_is_F_MPI_BOTTOM(sendbuf) ? MPI_BOTTOM : sendbuf;
   recvbuf = (void*) vftr_is_F_MPI_IN_PLACE(recvbuf) ? MPI_IN_PLACE : recvbuf;
   recvbuf = (void*) vftr_is_F_MPI_BOTTOM(recvbuf) ? MPI_BOTTOM : recvbuf;

   int c_error;
   if (isintercom) {
      c_error = vftr_MPI_Iscatterv_intercom(sendbuf,
                                            c_sendcounts,
                                            c_displs,
                                            c_sendtype,
                                            recvbuf,
                                            (int)(*recvcount),
                                            c_recvtype,
                                            (int)(*root),
                                            c_comm,
                                            &c_request);
   } else {
      if (vftr_is_C_MPI_IN_PLACE(recvbuf)) {
         c_error = vftr_MPI_Iscatterv_inplace(sendbuf,
                                              c_sendcounts,
                                              c_displs,
                                              c_sendtype,
                                              recvbuf,
                                              (int)(*recvcount),
                                              c_recvtype,
                                              (int)(*root),
                                              c_comm,
                                              &c_request);
      } else {
         c_error = vftr_MPI_Iscatterv(sendbuf,
                                      c_sendcounts,
                                      c_displs,
                                      c_sendtype,
                                      recvbuf,
                                      (int)(*recvcount),
                                      c_recvtype,
                                      (int)(*root),
                                      c_comm,
                                      &c_request);
      }
   }

   *f_error = (MPI_Fint) (c_error);
   *f_request = PMPI_Request_c2f(c_request);
}

#endif
