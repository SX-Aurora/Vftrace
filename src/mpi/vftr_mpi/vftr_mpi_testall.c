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

#include "vftr_mpi_pcontrol.h"
#include "vftr_clear_requests.h"
  
int vftr_MPI_Testall(int count, MPI_Request array_of_requests[],
                     int *flag, MPI_Status array_of_statuses[]) {
   int retVal;
   int tmpflag;

   // set the return flag to true
   *flag = true;
   // It will be returned true if all communications are completed
   for (int ireq=0; ireq<count; ireq++) {
      if (array_of_statuses == MPI_STATUSES_IGNORE) {
         retVal = PMPI_Request_get_status(array_of_requests[ireq],
                                          &tmpflag,
                                          MPI_STATUS_IGNORE);
      } else {
         retVal = PMPI_Request_get_status(array_of_requests[ireq],
                                          &tmpflag,
                                          array_of_statuses+ireq);
      }
      // accumulate truthness of the individual requests
      // as soon as one if false everything is
      *flag = (*flag) && tmpflag;
   }
   // clear completed communications from the list of open requests
   vftr_clear_completed_requests();

   if (flag) {
   // If all communications are completed
   // run Testall to modify the requests appropriately
      retVal = PMPI_Testall(count, array_of_requests, flag, array_of_statuses);
   }

   return retVal;
}

#endif
