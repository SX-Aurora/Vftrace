#ifdef _MPI
#include <mpi.h>

#include <stdlib.h>

#include "test.h"

void vftr_MPI_Test_f2vftr(MPI_Fint *f_request, MPI_Fint *f_flag,
                          MPI_Fint *f_status, MPI_Fint *f_error) {

   MPI_Request c_request;
   c_request = PMPI_Request_f2c(*f_request);
   int c_flag;
   MPI_Status c_status;

   int c_error = vftr_MPI_Test(&c_request,
                               &c_flag,
                               &c_status);

   *f_request = PMPI_Request_c2f(c_request);
   *f_flag = (MPI_Fint) c_flag;
   if (f_status != MPI_F_STATUS_IGNORE) {
      PMPI_Status_c2f(&c_status, f_status);
   }
   *f_error = (MPI_Fint) c_error;
}

#endif
