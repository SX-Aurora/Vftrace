#ifdef _MPI
#include <mpi.h>

#include <stdlib.h>

#include "testsome.h"

void vftr_MPI_Testsome_f082vftr(MPI_Fint *f_incount, MPI_Fint *f_array_of_requests,
                                MPI_Fint *f_outcount, MPI_Fint *f_array_of_indices,
                                MPI_F08_status *f_array_of_statuses, MPI_Fint *f_error) {

   int c_incount = (int)(*f_incount);
   MPI_Request *c_array_of_requests = (MPI_Request*)
                                      malloc(c_incount*sizeof(MPI_Request));
   for (int ireq=0; ireq<c_incount; ireq++) {
      c_array_of_requests[ireq] = PMPI_Request_f2c(f_array_of_requests[ireq]);
   }
   int c_outcount;
   int *c_array_of_indices = (int*) malloc(c_incount*sizeof(int));
   MPI_Status *c_array_of_statuses = MPI_STATUSES_IGNORE;
   if (f_array_of_statuses != MPI_F08_STATUSES_IGNORE) {
      c_array_of_statuses = (MPI_Status*) malloc(c_incount*sizeof(MPI_Status));
   }

   int c_error = vftr_MPI_Testsome(c_incount,
                                   c_array_of_requests,
                                   &c_outcount,
                                   c_array_of_indices,
                                   c_array_of_statuses);

   for (int ireq=0; ireq<c_incount; ireq++) {
      f_array_of_requests[ireq] = PMPI_Request_c2f(c_array_of_requests[ireq]);
   }
   free(c_array_of_requests);
   *f_outcount = (MPI_Fint) c_outcount;
   for (int ireq=0; ireq<c_outcount; ireq++) {
      f_array_of_indices[ireq] = (MPI_Fint) (c_array_of_indices[ireq] + 1);
   }
   free(c_array_of_indices);
   if (f_array_of_statuses != MPI_F08_STATUSES_IGNORE) {
      for (int ireq=0; ireq<c_outcount; ireq++) {
         PMPI_Status_c2f08(c_array_of_statuses+ireq, f_array_of_statuses+ireq);
      }
      free(c_array_of_statuses);
   }
   *f_error = (MPI_Fint) c_error;
}

#endif
