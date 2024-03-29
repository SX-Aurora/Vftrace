#ifdef _MPI
#include <mpi.h>

#include <stdlib.h>

#include "mpi_buf_addr_const.h"
#include "gatherv.h"

void vftr_MPI_Gatherv_f082vftr(void *sendbuf, MPI_Fint *sendcount, MPI_Fint *f_sendtype,
                               void *recvbuf, MPI_Fint *f_recvcounts, MPI_Fint *f_displs,
                               MPI_Fint *f_recvtype, MPI_Fint *root, MPI_Fint *f_comm,
                               MPI_Fint *f_error) {

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

   int *c_recvcounts = NULL ;
   int *c_displs = NULL ;
   if (isroot) {
      if (isintercom) {
         PMPI_Comm_remote_size(c_comm, &size);
      } else {
         PMPI_Comm_size(c_comm, &size);
      }
      c_recvcounts = (int*) malloc(size*sizeof(int));
      for (int i=0; i<size; i++) {
         c_recvcounts[i] = (int) f_recvcounts[i];
      }
      c_displs = (int*) malloc(size*sizeof(int));
      for (int i=0; i<size; i++) {
         c_displs[i] = (int) f_displs[i];
      }
   }
   MPI_Datatype c_sendtype = PMPI_Type_f2c(*f_sendtype);
   MPI_Datatype c_recvtype = PMPI_Type_f2c(*f_recvtype);

   sendbuf = (void*) vftr_is_F_MPI_IN_PLACE(sendbuf) ? MPI_IN_PLACE : sendbuf;
   sendbuf = (void*) vftr_is_F_MPI_BOTTOM(sendbuf) ? MPI_BOTTOM : sendbuf;
   recvbuf = (void*) vftr_is_F_MPI_BOTTOM(recvbuf) ? MPI_BOTTOM : recvbuf;

   int c_error;
   if (isintercom) {
      c_error = vftr_MPI_Gatherv_intercom(sendbuf,
                                          (int)(*sendcount),
                                          c_sendtype,
                                          recvbuf,
                                          c_recvcounts,
                                          c_displs,
                                          c_recvtype,
                                          (int)(*root),
                                          c_comm);
   } else {
      if (vftr_is_C_MPI_IN_PLACE(sendbuf)) {
         c_error = vftr_MPI_Gatherv_inplace(sendbuf,
                                            (int)(*sendcount),
                                            c_sendtype,
                                            recvbuf,
                                            c_recvcounts,
                                            c_displs,
                                            c_recvtype,
                                            (int)(*root),
                                            c_comm);
      } else {
         c_error = vftr_MPI_Gatherv(sendbuf,
                                    (int)(*sendcount),
                                    c_sendtype,
                                    recvbuf,
                                    c_recvcounts,
                                    c_displs,
                                    c_recvtype,
                                    (int)(*root),
                                    c_comm);
      }
   }

   if (isroot) {
      free(c_recvcounts);
      free(c_displs);
   }

   *f_error = (MPI_Fint) (c_error);
}

#endif
