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

#include "mpi_logging.h"
#include "testany_c2vftr.h"

int MPI_Testany(int count, MPI_Request array_of_requests[],
                int *index, int *flag, MPI_Status *status) {
   if (vftr_no_mpi_logging()) {
      return PMPI_Testany(count, array_of_requests, index, flag, status);
   } else {
      return vftr_MPI_Testany_c2vftr(count, array_of_requests, index, flag, status);
   }
}

#endif
