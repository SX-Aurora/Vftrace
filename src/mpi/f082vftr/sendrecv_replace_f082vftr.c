#ifdef _MPI
#include <mpi.h>

#include "sendrecv_replace.h"

void vftr_MPI_Sendrecv_replace_f082vftr(void *buf, MPI_Fint *count,
                                        MPI_Fint *f_datatype, MPI_Fint *dest,
                                        MPI_Fint *sendtag, MPI_Fint *source,
                                        MPI_Fint *recvtag, MPI_Fint *f_comm,
                                        MPI_F08_status *f_status, MPI_Fint *f_error) {

   MPI_Comm c_comm = PMPI_Comm_f2c(*f_comm);
   MPI_Datatype c_datatype = PMPI_Type_f2c(*f_datatype);
   MPI_Status c_status;

   int c_error = vftr_MPI_Sendrecv_replace(buf,
                                           (int)(*count),
                                           c_datatype,
                                           (int)(*dest),
                                           (int)(*sendtag),
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
