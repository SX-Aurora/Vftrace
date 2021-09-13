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

#include "vftr_mpi_utils.h"
#include "testall.h"

int vftr_MPI_Testall_c2vftr(int count, MPI_Request array_of_requests[],
                            int *flag, MPI_Status array_of_statuses[]) {
   if (vftr_no_mpi_logging()) {
      return PMPI_Testall(count, array_of_requests, flag, array_of_statuses);
   } else {
      return vftr_MPI_Testall(count, array_of_requests, flag, array_of_statuses);
   }
}

#endif