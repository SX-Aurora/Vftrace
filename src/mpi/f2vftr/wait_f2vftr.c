#ifdef _MPI
#include <mpi.h>

#include <stdlib.h>

#include "wait.h"

void vftr_MPI_Wait_f2vftr(MPI_Fint *f_request, MPI_Fint *f_status, MPI_Fint *f_error) {

   MPI_Request c_request;
   c_request = PMPI_Request_f2c(*f_request);
   MPI_Status c_status;

   int c_error = vftr_MPI_Wait(&c_request,
                               &c_status);

   *f_request = PMPI_Request_c2f(c_request);
   if (f_status != MPI_F_STATUS_IGNORE) {
      PMPI_Status_c2f(&c_status, f_status);
   }
   *f_error = (MPI_Fint) c_error;
}

#endif
