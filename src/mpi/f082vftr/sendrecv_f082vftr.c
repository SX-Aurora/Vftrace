#ifdef _MPI
#include <mpi.h>

#include "sendrecv.h"

void vftr_MPI_Sendrecv_f082vftr(void *sendbuf, MPI_Fint *sendcount,
                                MPI_Fint *f_sendtype, MPI_Fint *dest, MPI_Fint *sendtag,
                                void *recvbuf, MPI_Fint *recvcount, MPI_Fint *f_recvtype,
                                MPI_Fint *source, MPI_Fint *recvtag, MPI_Fint *f_comm,
                                MPI_F08_status *f_status, MPI_Fint *f_error) {

   MPI_Comm c_comm = PMPI_Comm_f2c(*f_comm);
   MPI_Datatype c_sendtype = PMPI_Type_f2c(*f_sendtype);
   MPI_Datatype c_recvtype = PMPI_Type_f2c(*f_recvtype);
   MPI_Status c_status;

   int c_error = vftr_MPI_Sendrecv(sendbuf,
                                   (int)(*sendcount),
                                   c_sendtype,
                                   (int)(*dest),
                                   (int)(*sendtag),
                                   recvbuf,
                                   (int)(*recvcount),
                                   c_recvtype,
                                   (int)(*source),
                                   (int)(*recvtag),
                                   c_comm,
                                   &c_status);

   if (f_status != MPI_F08_STATUS_IGNORE) {
      PMPI_Status_c2f08(&c_status, f_status);
   }
   *f_error = (MPI_Fint) (c_error);
}

#endif
