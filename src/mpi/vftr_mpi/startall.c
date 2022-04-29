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

#include "vftr_timer.h"
#include "requests.h"

int vftr_MPI_Startall(int count, MPI_Request *array_of_requests) {
   long long tstart = vftr_get_runtime_usec();
   int retVal = PMPI_Startall(count, array_of_requests);

   long long t2start = vftr_get_runtime_usec();
   for (int ireq=0; ireq<count; ireq++) {
      vftr_activate_pers_request(array_of_requests[ireq], tstart) ;
   }
   long long t2end = vftr_get_runtime_usec();

   vftr_mpi_overhead_usec += t2end - t2start;

   return retVal;
}
