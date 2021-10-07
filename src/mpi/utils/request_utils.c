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

#include <stdbool.h>

#include <mpi.h>

#include "status_utils.h"

// check if a request is active
bool vftr_mpi_request_is_active(MPI_Request request) {
   // According to the MPI_Standard 3.0 (capter 3.7.3, p.52)
   // a request is active if it is neither a null request
   // nor returns an empty status for Request_get_status 
   // (the function returns an empty status if it is inactive)
   
   if (request == MPI_REQUEST_NULL) {
      return false;
   }

   MPI_Status status;
   int flag;
   PMPI_Request_get_status(request, &flag, &status);

   return !vftr_mpi_status_is_empty(&status);
}
