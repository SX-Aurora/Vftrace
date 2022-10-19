#ifdef _MPI
#include <mpi.h>

#include "mpi_buf_addr_const.h"
#include "ibcast.h"

void vftr_MPI_Ibcast_f082vftr(void *buffer, MPI_Fint *count, MPI_Fint *f_datatype,
                              MPI_Fint *root, MPI_Fint *f_comm, MPI_Fint *f_request,
                              MPI_Fint *f_error) {

   MPI_Datatype c_datatype = PMPI_Type_f2c(*f_datatype);
   MPI_Comm c_comm = PMPI_Comm_f2c(*f_comm);
   MPI_Request c_request;

   int c_error;
   int isintercom;
   PMPI_Comm_test_inter(c_comm, &isintercom);
   if (isintercom) {
      c_error = vftr_MPI_Ibcast_intercom(buffer,
                                         (int)(*count),
                                         c_datatype,
                                         (int)(*root),
                                         c_comm,
                                         &c_request);
   } else {
      c_error = vftr_MPI_Ibcast(buffer,
                                (int)(*count),
                                c_datatype,
                                (int)(*root),
                                c_comm,
                                &c_request);
   }

   *f_error = (MPI_Fint) (c_error);
   *f_request = PMPI_Request_c2f(c_request);
}

#endif
