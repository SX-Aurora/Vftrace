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

#include "vftr_mpi_pcontrol.h"
#include "vftr_async_messages.h"
  
int vftr_MPI_Waitall(int count, MPI_Request array_of_requests[],
                     MPI_Status array_of_statuses[]) {

   // disable profiling based on the Pcontrol level
   if (vftrace_Pcontrol_level == 0) {
      return PMPI_Waitall(count, array_of_requests, array_of_statuses);
   } else {
      if (count <= 0) {return MPI_SUCCESS;}
   
      // loop while at least one request is not completed
      int *req_completed = (int*) malloc(count*sizeof(int));
      for (int ireq=0; ireq<count; ireq++) {
         req_completed[ireq] = false;
      }
      int retVal;
      int tmpflag;
      bool all_completed = false;
      while (!all_completed) {
         all_completed = true;
         // loop over all requests
         for (int ireq=0; ireq<count; ireq++) {
            if (!req_completed[ireq]) {
               // check if the communication associated with the request
               // is completed
               PMPI_Request_get_status(array_of_requests[ireq],
                                       &tmpflag,
                                       MPI_STATUS_IGNORE);
               // if not completed 
               req_completed[ireq] = tmpflag;
               if (!(req_completed[ireq])) {
                  all_completed = false;
               }
            }
         }
         vftr_clear_completed_request();
      }

      free(req_completed);
      req_completed = NULL;
   
      return PMPI_Waitall(count, array_of_requests, array_of_statuses);
   }
}

#endif
