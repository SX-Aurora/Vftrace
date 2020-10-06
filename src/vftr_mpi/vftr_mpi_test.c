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
  
int vftr_MPI_Test(MPI_Request *request, int *flag, MPI_Status *status) {

   // disable profiling based on the Pcontrol level
   if (vftrace_Pcontrol_level == 0) {
      return PMPI_Test(request, flag, status);
   } else {
      // do not call MPI_Test immediately!
      // If the communication is successfull MPI_Test destroys the Reqeust 
      // Hence, no chance of properly clearing the communication
      // from the open request list
      // MPI_Request_get_status is a non destructive check. 
      int retVal = PMPI_Request_get_status(*request, flag, status);
   
      if (*flag) {
         // Communication is done.
         // Clear finished communications from the open request list
         vftr_clear_completed_request();
         vftr_clear_completed_requests();
         // Now that the danger of deleating needed requests is banned
         // actually call MPI_Test   
         retVal = PMPI_Test(request, flag, status);
      }
   
      return retVal;
   }
}

#endif
