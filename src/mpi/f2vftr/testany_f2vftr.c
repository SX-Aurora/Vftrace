#ifdef _MPI
#include <mpi.h>

#include <stdlib.h>

#include "testany.h"

void vftr_MPI_Testany_f2vftr(MPI_Fint *f_count, MPI_Fint *f_array_of_requests,
                             MPI_Fint *f_index, MPI_Fint *f_flag, MPI_Fint *f_status,
                             MPI_Fint *f_error) {

   int c_count = (int)(*f_count);
   MPI_Request *c_array_of_requests = (MPI_Request*) malloc(c_count*sizeof(MPI_Request));
   for (int ireq=0; ireq<c_count; ireq++) {
      c_array_of_requests[ireq] = PMPI_Request_f2c(f_array_of_requests[ireq]);
   }
   int c_index;
   int c_flag;
   MPI_Status c_status;

   int c_error = vftr_MPI_Testany(c_count,
                                  c_array_of_requests,
                                  &c_index,
                                  &c_flag,
                                  &c_status);

   for (int ireq=0; ireq<c_count; ireq++) {
      f_array_of_requests[ireq] = PMPI_Request_c2f(c_array_of_requests[ireq]);
   }
   free(c_array_of_requests);
   *f_index = (MPI_Fint) (c_index+1);
   *f_flag = (MPI_Fint) c_flag;
   if (f_status != MPI_F_STATUS_IGNORE) {
      PMPI_Status_c2f(&c_status, f_status);
   }
   *f_error = (MPI_Fint) c_error;
}

#endif
