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
#include "mpiprofiling.h"

int vftr_MPI_Init_thread(int *argc, char ***argv,
                         int required, int *provided) {
   SELF_PROFILE_START_FUNCTION;
   int retVal = PMPI_Init_thread(argc, argv, required, provided);

   PMPI_Comm_size(MPI_COMM_WORLD, &vftrace.process.nprocesses);
   PMPI_Comm_rank(MPI_COMM_WORLD, &vftrace.process.processID);

   vftr_create_profiled_ranks_list(vftrace.environment,
                                   vftrace.process,
                                   &vftrace.mpi_state);
   SELF_PROFILE_END_FUNCTION;
   return retVal;
}
