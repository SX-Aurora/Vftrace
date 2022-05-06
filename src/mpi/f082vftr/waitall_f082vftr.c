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

#include "waitall.h"

void vftr_MPI_Waitall_f082vftr(MPI_Fint *f_count, MPI_Fint *f_array_of_requests,
                               MPI_F08_status *f_array_of_statuses, MPI_Fint *f_error) {

   int c_count = (int)(*f_count);
   MPI_Request *c_array_of_requests = (MPI_Request*)
                                      malloc(c_count*sizeof(MPI_Request));
   for (int ireq=0; ireq<c_count; ireq++) {
      c_array_of_requests[ireq] = PMPI_Request_f2c(f_array_of_requests[ireq]);
   }
   MPI_Status *c_array_of_statuses = MPI_STATUSES_IGNORE;
   if (f_array_of_statuses != MPI_F08_STATUSES_IGNORE) {
      c_array_of_statuses = (MPI_Status*) malloc(c_count*sizeof(MPI_Status));
   }

   int c_error = vftr_MPI_Waitall(c_count,
                                  c_array_of_requests,
                                  c_array_of_statuses);

   for (int ireq=0; ireq<c_count; ireq++) {
      f_array_of_requests[ireq] = PMPI_Request_c2f(c_array_of_requests[ireq]);
   }
   free(c_array_of_requests);
   if (f_array_of_statuses != MPI_F08_STATUSES_IGNORE) {
      for (int ireq=0; ireq<c_count; ireq++) {
         PMPI_Status_c2f08(c_array_of_statuses+ireq, f_array_of_statuses+ireq);
      }
      free(c_array_of_statuses);
   }
   *f_error = (MPI_Fint) c_error;
}

#endif
