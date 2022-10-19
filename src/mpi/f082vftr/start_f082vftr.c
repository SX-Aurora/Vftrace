#ifdef _MPI
#include <mpi.h>

#include "start.h"

void vftr_MPI_Start_f082vftr(MPI_Fint *f_request, MPI_Fint *f_error) {

   MPI_Request c_request;
   c_request = PMPI_Request_f2c(*f_request);

   int c_error = vftr_MPI_Start(&c_request);

   *f_request = PMPI_Request_c2f(c_request);
   *f_error = (MPI_Fint) c_error;
}

#endif
