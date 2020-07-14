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

#include <stdlib.h>

#ifdef _MPI
#include <mpi.h>

#include "vftr_mpi_testwait.h"
  
void vftr_MPI_Barrier_F(MPI_Fint *f_comm, MPI_Fint *f_error) {

   MPI_Comm c_comm = PMPI_Comm_f2c(*f_comm);

   int c_error = vftr_MPI_Barrier(c_comm);

   *f_error = (MPI_Fint) c_error;
}

void vftr_MPI_Test_F(MPI_Fint *f_request, MPI_Fint *f_flag,
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
      MPI_Status_c2f(&c_status, f_status);
   }
   *f_error = (MPI_Fint) c_error;
}

void vftr_MPI_Testany_F(MPI_Fint *f_count, MPI_Fint *f_array_of_requests,
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
      MPI_Status_c2f(&c_status, f_status);
   }
   *f_error = (MPI_Fint) c_error;
}

void vftr_MPI_Testsome_F(MPI_Fint *f_incount, MPI_Fint *f_array_of_requests,
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
   if (f_array_of_statuses != MPI_F_STATUSES_IGNORE) {
      for (int ireq=0; ireq<c_outcount; ireq++) {
         MPI_Status_c2f(c_array_of_statuses+ireq, f_array_of_statuses+ireq);
      }
      free(c_array_of_statuses);
   }
   *f_error = (MPI_Fint) c_error;
}

void vftr_MPI_Testall_F(MPI_Fint *f_count, MPI_Fint *f_array_of_requests,
                        MPI_Fint *f_flag, MPI_Fint *f_array_of_statuses,
                        MPI_Fint *f_error) {

   int c_count = (int)(*f_count);
   MPI_Request *c_array_of_requests = (MPI_Request*)
                                      malloc(c_count*sizeof(MPI_Request));
   for (int ireq=0; ireq<c_count; ireq++) {
      c_array_of_requests[ireq] = PMPI_Request_f2c(f_array_of_requests[ireq]);
   }
   int c_flag;
   MPI_Status *c_array_of_statuses = MPI_STATUSES_IGNORE;
   if (f_array_of_statuses != MPI_F_STATUSES_IGNORE) {
      c_array_of_statuses = (MPI_Status*) malloc(c_count*sizeof(MPI_Status));
   }

   int c_error = vftr_MPI_Testall(c_count,
                                  c_array_of_requests,
                                  &c_flag,
                                  c_array_of_statuses);

   for (int ireq=0; ireq<c_count; ireq++) {
      f_array_of_requests[ireq] = PMPI_Request_c2f(c_array_of_requests[ireq]);
   }
   free(c_array_of_requests);
   *f_flag = (MPI_Fint) c_flag;
   if (f_array_of_statuses != MPI_F_STATUSES_IGNORE) {
      for (int ireq=0; ireq<c_count; ireq++) {
         MPI_Status_c2f(c_array_of_statuses+ireq, f_array_of_statuses+ireq);
      }
      free(c_array_of_statuses);
   }
   *f_error = (MPI_Fint) c_error;
}

void vftr_MPI_Wait_F(MPI_Fint *f_request, MPI_Fint *f_status, MPI_Fint *f_error) {

   MPI_Request c_request;
   c_request = PMPI_Request_f2c(*f_request);
   MPI_Status c_status;

   int c_error = vftr_MPI_Wait(&c_request,
                               &c_status);

   *f_request = PMPI_Request_c2f(c_request);
   if (f_status != MPI_F_STATUS_IGNORE) {
      MPI_Status_c2f(&c_status, f_status);
   }
   *f_error = (MPI_Fint) c_error;
}

void vftr_MPI_Waitany_F(MPI_Fint *f_count, MPI_Fint *f_array_of_requests,
                        MPI_Fint *f_index, MPI_Fint *f_status, MPI_Fint *f_error) {

   int c_count = (int)(*f_count);
   MPI_Request *c_array_of_requests = (MPI_Request*) malloc(c_count*sizeof(MPI_Request));
   for (int ireq=0; ireq<c_count; ireq++) {
      c_array_of_requests[ireq] = PMPI_Request_f2c(f_array_of_requests[ireq]);
   }
   int c_index;
   MPI_Status c_status;

   int c_error = vftr_MPI_Waitany(c_count,
                                  c_array_of_requests,
                                  &c_index,
                                  &c_status);

   for (int ireq=0; ireq<c_count; ireq++) {
      f_array_of_requests[ireq] = PMPI_Request_c2f(c_array_of_requests[ireq]);
   }
   free(c_array_of_requests);
   *f_index = (MPI_Fint) (c_index+1);
   if (f_status != MPI_F_STATUS_IGNORE) {
      MPI_Status_c2f(&c_status, f_status);
   }
   *f_error = (MPI_Fint) c_error;
}

void vftr_MPI_Waitsome_F(MPI_Fint *f_incount, MPI_Fint *f_array_of_requests,
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
         MPI_Status_c2f(c_array_of_statuses+ireq, f_array_of_statuses+ireq);
      }
      free(c_array_of_statuses);
   }
   *f_error = (MPI_Fint) c_error;
}

void vftr_MPI_Waitall_F(MPI_Fint *f_count, MPI_Fint *f_array_of_requests,
                        MPI_Fint *f_array_of_statuses, MPI_Fint *f_error) {

   int c_count = (int)(*f_count);
   MPI_Request *c_array_of_requests = (MPI_Request*)
                                      malloc(c_count*sizeof(MPI_Request));
   for (int ireq=0; ireq<c_count; ireq++) {
      c_array_of_requests[ireq] = PMPI_Request_f2c(f_array_of_requests[ireq]);
   }
   MPI_Status *c_array_of_statuses = MPI_STATUSES_IGNORE;
   if (f_array_of_statuses != MPI_F_STATUSES_IGNORE) {
      c_array_of_statuses = (MPI_Status*) malloc(c_count*sizeof(MPI_Status));
   }

   int c_error = vftr_MPI_Waitall(c_count,
                                  c_array_of_requests,
                                  c_array_of_statuses);

   for (int ireq=0; ireq<c_count; ireq++) {
      f_array_of_requests[ireq] = PMPI_Request_c2f(c_array_of_requests[ireq]);
   }
   free(c_array_of_requests);
   if (f_array_of_statuses != MPI_F_STATUSES_IGNORE) {
      for (int ireq=0; ireq<c_count; ireq++) {
         MPI_Status_c2f(c_array_of_statuses+ireq, f_array_of_statuses+ireq);
      }
      free(c_array_of_statuses);
   }
   *f_error = (MPI_Fint) c_error;
}

#endif
