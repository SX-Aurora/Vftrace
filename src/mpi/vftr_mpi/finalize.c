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

#include "self_profile.h"
#include "vftrace_state.h"
#include "vftr_finalize.h"
#include "mpiprofiling.h"
#include "requests.h"

int vftr_MPI_Finalize() {
   SELF_PROFILE_START_FUNCTION;
   vftr_free_request_list(&vftrace.mpi_state);
   vftr_free_profiled_ranks_list(&vftrace.mpi_state);
   SELF_PROFILE_END_FUNCTION;
   // it is neccessary to finalize vftrace here, in order to properly communicat stack ids
   // between processes. After MPI_Finalize communication between processes is prohibited
   vftr_finalize();

   return PMPI_Finalize();
}
