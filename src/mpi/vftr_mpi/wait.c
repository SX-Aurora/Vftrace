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

#include <mpi.h>

#include <stdbool.h>

#include "clear_mpi_requests.h"
  
int vftr_MPI_Wait(MPI_Request *request, MPI_Status *status) {
   int retVal;

   // loop until the communication corresponding to the request is completed
   int flag = false;
   while (!flag) {
      // check if the communication is finished
      retVal = PMPI_Request_get_status(*request,
                                       &flag,
                                       status);
      // either the communication is completed, or not
      // other communications might be completed in the background
      // clear those from the list of open requests
      vftr_clear_completed_requests();
   }
   // Properly set the request and status variable
   retVal = PMPI_Wait(request, status);

   return retVal;
}
