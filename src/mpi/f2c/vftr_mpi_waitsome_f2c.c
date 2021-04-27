/*
   This file is part of Vftrace.

   Vftrace is free software; you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation; either version 2 of the License, or
   (at your option) any later version.

   Vftrace is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License along
   with this program; if not, write to the Free Software Foundation, Inc.,
   51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
*/

#ifdef _MPI
#include <mpi.h>

#include <stdlib.h>

#include <vftr_mpi_waitsome.h>
  
void vftr_MPI_Waitsome_f2c(MPI_Fint *f_incount, MPI_Fint *f_array_of_requests,
                           MPI_Fint *f_outcount, MPI_Fint *f_array_of_indices,
                           MPI_Fint *f_array_of_statuses, MPI_Fint *f_error) {

   int c_incount = (int)(*f_incount);
   MPI_Request *c_array_of_requests = (MPI_Request*)
                                      malloc(c_incount*sizeof(MPI_Request));
   for (int ireq=0; ireq<c_incount; ireq++) {
      c_array_of_requests[ireq] = PMPI_Request_f2c(f_array_of_requests[ireq]);
   }
   int c_outcount;
   int *c_array_of_indices = (int*) malloc(c_incount*sizeof(int));
   MPI_Status *c_array_of_statuses = MPI_STATUSES_IGNORE;
   if (f_array_of_statuses != MPI_F_STATUSES_IGNORE) {
      c_array_of_statuses = (MPI_Status*) malloc(c_incount*sizeof(MPI_Status));
   }

   int c_error = vftr_MPI_Waitsome(c_incount,
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
   if (f_array_of_statuses != MPI_F_STATUSES_IGNORE) {
      for (int ireq=0; ireq<c_outcount; ireq++) {
         PMPI_Status_c2f(c_array_of_statuses+ireq, f_array_of_statuses+ireq);
      }
      free(c_array_of_statuses);
   }
   *f_error = (MPI_Fint) c_error;
}

#endif
