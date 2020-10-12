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

#include "vftr_timer.h"
#include "vftr_persistent_requests.h"
#include "vftr_p2p_requests.h"
#include "vftr_mpi_utils.h"

int vftr_MPI_Startall(int count, MPI_Request *array_of_requests) {

   // disable profiling based on the Pcontrol level
   if (vftr_no_mpi_logging()) {
      return PMPI_Startall(count, array_of_requests);
   } else {
      long long tstart = vftr_get_runtime_usec();
      int retVal = PMPI_Startall(count, array_of_requests);

      long long t2start = vftr_get_runtime_usec();
      for (int ireq=0; ireq<count; ireq++) {
         vftr_request_t *matching_persistent_request = vftr_search_request(vftr_open_persistent_request_list,
                                                                           array_of_requests[ireq]);
         vftr_register_P2P_request(matching_persistent_request->dir, 
                                   matching_persistent_request->count[0],
                                   matching_persistent_request->type[0],
                                   matching_persistent_request->rank[0],
                                   matching_persistent_request->tag,
                                   matching_persistent_request->comm,
                                   matching_persistent_request->request,
                                   tstart);
      }

      long long t2end = vftr_get_runtime_usec();

      vftr_mpi_overhead_usec += t2end - t2start;

      return retVal;
   }
}

#endif
