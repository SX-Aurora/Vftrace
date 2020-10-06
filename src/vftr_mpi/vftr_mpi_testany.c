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

#include "vftr_mpi_pcontrol.h"
#include "vftr_async_messages.h"
#include "vftr_requests.h"
  
int vftr_MPI_Testany(int count, MPI_Request array_of_requests[],
                     int *index, int *flag, MPI_Status *status) {

   // disable profiling based on the Pcontrol level
   if (vftrace_Pcontrol_level == 0) {
      return PMPI_Testany(count, array_of_requests, index, flag, status);
   } else {
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
         *flag = true;
         *index = MPI_UNDEFINED;
         if (status != MPI_STATUS_IGNORE) {
            vftr_empty_mpi_status(status);
         }
         return MPI_SUCCESS;
      }

      // initialize the index to the default failure value
      *index = MPI_UNDEFINED;
      // loop over all requests and terminate the loop
      // on the first completed communication
      int retVal = 0;
      for (int ireq=0; ireq<count; ireq++) {
         retVal = PMPI_Request_get_status(array_of_requests[ireq],
                                          flag,
                                          status);
         // terminate if a completed communication is found
         if (flag) {
            // record the index of the completed communication
            *index = ireq;
            // clear completed communication from the list of open requests
            vftr_clear_completed_request();
            vftr_clear_completed_requests();
            // Mark the request as inactive, or deallocate it.
            retVal = PMPI_Test(array_of_requests+ireq,
                               flag,
                               status);
            break;
         }
      }

      return retVal;
   }
}

#endif
