#ifdef _MPI
#include <mpi.h>

#include <stdlib.h>

#include "ibarrier.h"

void vftr_MPI_Ibarrier_f082vftr(MPI_Fint *f_comm, MPI_Fint *f_request,
                                MPI_Fint *f_error) {

   MPI_Comm c_comm = PMPI_Comm_f2c(*f_comm);
   MPI_Request c_request;

   int c_error = vftr_MPI_Ibarrier(c_comm,
                                   &c_request);

   *f_request = PMPI_Request_c2f(c_request);
   *f_error = (MPI_Fint) c_error;
}

#endif
