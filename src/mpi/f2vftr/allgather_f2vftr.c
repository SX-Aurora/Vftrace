#ifdef _MPI
#include <mpi.h>

#include "mpi_buf_addr_const.h"
#include "allgather.h"

void vftr_MPI_Allgather_f2vftr(void *sendbuf, MPI_Fint *sendcount, MPI_Fint *f_sendtype,
                               void *recvbuf, MPI_Fint *recvcount, MPI_Fint *f_recvtype,
                               MPI_Fint *f_comm, MPI_Fint *f_error) {

   MPI_Datatype c_sendtype = PMPI_Type_f2c(*f_sendtype);
   MPI_Datatype c_recvtype = PMPI_Type_f2c(*f_recvtype);
   MPI_Comm c_comm = PMPI_Comm_f2c(*f_comm);

   sendbuf = (void*) vftr_is_F_MPI_IN_PLACE(sendbuf) ? MPI_IN_PLACE : sendbuf;
   sendbuf = (void*) vftr_is_F_MPI_BOTTOM(sendbuf) ? MPI_BOTTOM : sendbuf;
   recvbuf = (void*) vftr_is_F_MPI_BOTTOM(recvbuf) ? MPI_BOTTOM : recvbuf;

   int c_error;
   int isintercom;
   PMPI_Comm_test_inter(c_comm, &isintercom);
   if (isintercom) {
      c_error = vftr_MPI_Allgather_intercom(sendbuf,
                                            (int)(*sendcount),
                                            c_sendtype,
                                            recvbuf,
                                            (int)(*recvcount),
                                            c_recvtype,
                                            c_comm);
   } else {
      if (vftr_is_C_MPI_IN_PLACE(sendbuf)) {
         c_error = vftr_MPI_Allgather_inplace(sendbuf,
                                              (int)(*sendcount),
                                              c_sendtype,
                                              recvbuf,
                                              (int)(*recvcount),
                                              c_recvtype,
                                              c_comm);
      } else {
         c_error = vftr_MPI_Allgather(sendbuf,
                                      (int)(*sendcount),
                                      c_sendtype,
                                      recvbuf,
                                      (int)(*recvcount),
                                      c_recvtype,
                                      c_comm);
      }
   }

   *f_error = (MPI_Fint) (c_error);
}

#endif
