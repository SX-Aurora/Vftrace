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

#include <stdbool.h>

#include "vftr_mpi_utils.h"
#include "vftr_clear_requests.h"
  
int vftr_MPI_Waitany(int count, MPI_Request array_of_requests[],
                     int *index, MPI_Status *status) {
   if (count <= 0) {return MPI_SUCCESS;}
   
   // First check if the request array contains at least one active handle
   bool activereqs = false;
   for (int ireq=0; ireq<count; ireq++) {
      if (vftr_mpi_request_is_active(array_of_requests[ireq])) {
         activereqs = true;
         break;
      }
   }
   // if no active request is found return with the following settings
   if (!activereqs) {
      *index = MPI_UNDEFINED;
      if (status == MPI_STATUS_IGNORE) {
         vftr_empty_mpi_status(status);
      }
      return MPI_SUCCESS;
   }
   
   // loop until at least one communication of the requests is completed
   int retVal;
   int completed_req = false;
   while (!completed_req) {
      // loop over all requests
      for (int ireq=0; ireq<count; ireq++) {
         int flag;
         // check if the communication associated with the request
         // is completed
         retVal = PMPI_Request_get_status(array_of_requests[ireq],
                                          &flag,
                                          status);
         completed_req = completed_req || flag;
         // either the communication is completed, or not
         // other communications might be completed in the background
         // clear those from the list of open requests
         vftr_clear_completed_requests();
         // if this request corresponds to a completed communication
         // leave the loop
         if (flag) {
            // record the index of the finished request
            *index = ireq;
            break;
         }
      }
   }
   
   // Properly set the request and status variable
   retVal = PMPI_Wait(array_of_requests+(*index), status);
   
   return retVal;
}

#endif
