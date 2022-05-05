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

#include "timer.h"
#include "requests.h"
#include "p2p_requests.h"
#include "requests.h"

int vftr_MPI_Request_free(MPI_Request *request) {

   long long t2start= vftr_get_runtime_usec();
   vftr_request_t *matched_request = vftr_search_request(*request);
   if (matched_request != NULL) {
         matched_request->marked_for_deallocation = true;
   } else {
      PMPI_Request_free(request);
   }
   *request = MPI_REQUEST_NULL;
   long long t2end = vftr_get_runtime_usec();

   //TODO: vftr_mpi_overhead_usec += t2end - t2start;

   return 0;
}
