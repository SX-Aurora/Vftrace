#ifdef _MPI
#include <mpi.h>

#include <stdlib.h>

#include "startall.h"

void vftr_MPI_Startall_f2vftr(MPI_Fint *f_count, MPI_Fint *f_array_of_requests,
                              MPI_Fint *f_error) {

   int c_count = (int)(*f_count);
   MPI_Request *c_array_of_requests = (MPI_Request*)
                                      malloc(c_count*sizeof(MPI_Request));
   for (int ireq=0; ireq<c_count; ireq++) {
      c_array_of_requests[ireq] = PMPI_Request_f2c(f_array_of_requests[ireq]);
   }

   int c_error = vftr_MPI_Startall(c_count,
                                   c_array_of_requests);

   for (int ireq=0; ireq<c_count; ireq++) {
      f_array_of_requests[ireq] = PMPI_Request_c2f(c_array_of_requests[ireq]);
   }
   free(c_array_of_requests);
   *f_error = (MPI_Fint) c_error;
}

#endif
