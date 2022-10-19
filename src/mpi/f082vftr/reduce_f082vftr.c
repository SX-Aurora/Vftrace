#ifdef _MPI
#include <mpi.h>

#include "mpi_buf_addr_const.h"
#include "reduce.h"

void vftr_MPI_Reduce_f082vftr(void *sendbuf, void *recvbuf, MPI_Fint *count,
                              MPI_Fint *f_datatype, MPI_Fint *f_op, MPI_Fint *root,
                              MPI_Fint *f_comm, MPI_Fint *f_error) {

   MPI_Datatype c_datatype = PMPI_Type_f2c(*f_datatype);
   MPI_Op c_op = PMPI_Op_f2c(*f_op);
   MPI_Comm c_comm = PMPI_Comm_f2c(*f_comm);

   sendbuf = (void*) vftr_is_F_MPI_IN_PLACE(sendbuf) ? MPI_IN_PLACE : sendbuf;
   sendbuf = (void*) vftr_is_F_MPI_BOTTOM(sendbuf) ? MPI_BOTTOM : sendbuf;
   recvbuf = (void*) vftr_is_F_MPI_BOTTOM(recvbuf) ? MPI_BOTTOM : recvbuf;

   int c_error;
   int isintercom;
   PMPI_Comm_test_inter(c_comm, &isintercom);
   if (isintercom) {
      c_error = vftr_MPI_Reduce_intercom(sendbuf,
                                         recvbuf,
                                         (int)(*count),
                                         c_datatype,
                                         c_op,
                                         (int)(*root),
                                         c_comm);
   } else {
      if (vftr_is_C_MPI_IN_PLACE(sendbuf)) {
         c_error = vftr_MPI_Reduce_inplace(sendbuf,
                                           recvbuf,
                                           (int)(*count),
                                           c_datatype,
                                           c_op,
                                           (int)(*root),
                                           c_comm);
      } else {
         c_error = vftr_MPI_Reduce(sendbuf,
                                   recvbuf,
                                   (int)(*count),
                                   c_datatype,
                                   c_op,
                                   (int)(*root),
                                   c_comm);
      }
   }

   *f_error = (MPI_Fint) (c_error);
}

#endif
