#ifdef _MPI
#include <mpi.h>

#include "recv_init.h"

void vftr_MPI_Recv_init_f2vftr(void *buf, MPI_Fint *count, MPI_Fint *f_datatype,
                               MPI_Fint *source, MPI_Fint *tag, MPI_Fint *f_comm,
                               MPI_Fint *f_request, MPI_Fint *f_error) {

   MPI_Comm c_comm = PMPI_Comm_f2c(*f_comm);
   MPI_Datatype c_datatype = PMPI_Type_f2c(*f_datatype);
   MPI_Request c_request;

   int c_error = vftr_MPI_Recv_init(buf,
                                    (int)(*count),
                                    c_datatype,
                                    (int)(*source),
                                    (int)(*tag),
                                    c_comm,
                                    &c_request);

   *f_error = (MPI_Fint) (c_error);
   *f_request = PMPI_Request_c2f(c_request);
}

#endif
