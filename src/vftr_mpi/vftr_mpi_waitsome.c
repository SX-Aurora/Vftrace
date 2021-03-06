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
#include <stdbool.h>

#include "vftr_mpi_utils.h"
#include "vftr_mpi_pcontrol.h"
#include "vftr_clear_requests.h"
  
int vftr_MPI_Waitsome(int incount, MPI_Request array_of_requests[],
                      int *outcount, int array_of_indices[],
                      MPI_Status array_of_statuses[]) {

   // disable profiling based on the Pcontrol level
   if (vftrace_Pcontrol_level == 0) {
      return PMPI_Waitsome(incount, array_of_requests, outcount,
                           array_of_indices, array_of_statuses);
   } else {
      if (incount <= 0) {
         outcount = 0;
         return MPI_SUCCESS;
      }

      // First check if the request array contains at least one active handle
      bool activereqs = false;
      for (int ireq=0; ireq<incount; ireq++) {
         if (vftr_mpi_request_is_active(array_of_requests[ireq])) {
            activereqs = true;
            break;
         }
      }
      // if no active request is found return with the following settings
      if (!activereqs) {
         *outcount = MPI_UNDEFINED;
         return MPI_SUCCESS;
      }

      int retVal = MPI_SUCCESS;
      int tmpretVal;
      *outcount = 0;
      // loop while outcount is 0
      while (*outcount == 0) {
         // loop over all requests and check for completion
         for (int ireq=0; ireq<incount; ireq++) {
            int flag;
            if (array_of_statuses == MPI_STATUSES_IGNORE) {
               tmpretVal = PMPI_Request_get_status(array_of_requests[ireq],
                                                   &flag,
                                                   MPI_STATUS_IGNORE);
            } else {
               tmpretVal = PMPI_Request_get_status(array_of_requests[ireq],
                                                   &flag,
                                                   array_of_statuses+ireq);
            }
            if (tmpretVal != MPI_SUCCESS) {
               // if something goes wrong inform the
               // user to check the status variable
               retVal = MPI_ERR_IN_STATUS;
            }
            if (flag) {
               // record completed communications for return
               (*outcount)++;
               array_of_indices[(*outcount)-1] = ireq;
               // remove completed communications from the list of open requests
               vftr_clear_completed_requests();
               // Mark the request as inactive, or deallocate it.
               if (array_of_statuses == MPI_STATUSES_IGNORE) {
                  tmpretVal = PMPI_Wait(array_of_requests+ireq,
                                        MPI_STATUS_IGNORE);
               } else {
                  tmpretVal = PMPI_Wait(array_of_requests+ireq,
                                        array_of_statuses+ireq);
               }
               if (tmpretVal != MPI_SUCCESS) {
                  // if something goes wrong inform the
                  // user to check the status variable
                  retVal = MPI_ERR_IN_STATUS;
               }
            }
         }
      }

      return retVal;
   }
}

#endif
