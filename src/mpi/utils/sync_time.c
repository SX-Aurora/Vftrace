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

#include "vftr_environment.h"
#include "vftr_regions.h"

void vftr_estimate_sync_time(char *routine_name, MPI_Comm comm) {
   if (vftr_environment.mpi_show_sync_time->value) {
      vftr_internal_region_begin(routine_name);
      PMPI_Barrier(comm);
      vftr_internal_region_end(routine_name);
   }
}
