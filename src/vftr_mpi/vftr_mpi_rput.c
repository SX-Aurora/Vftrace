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
#include "vftr_onesided_requests.h"
#include "vftr_mpi_utils.h"

int vftr_MPI_Rput(const void *origin_addr, int origin_count,
                  MPI_Datatype origin_datatype, int target_rank,
                  MPI_Aint target_disp, int target_count,
                  MPI_Datatype target_datatype, MPI_Win win, 
                  MPI_Request *request) {

   // disable profiling based on the Pcontrol level
   if (vftr_no_mpi_logging()) {
      return PMPI_Rput(origin_addr, origin_count, origin_datatype,
                       target_rank, target_disp, target_count,
                       target_datatype, win, request);
   } else {
      long long tstart = vftr_get_runtime_usec();
      int retVal = PMPI_Rput(origin_addr, origin_count, origin_datatype,
                             target_rank, target_disp, target_count,
                             target_datatype, win, request);

      long long t2start = vftr_get_runtime_usec();
      // Need to figure out the partner rank in a known communicator to store info
      MPI_Group local_group;
      PMPI_Win_get_group(win, &local_group);

      MPI_Group global_group;
      PMPI_Comm_group(MPI_COMM_WORLD, &global_group);

      int global_rank;
      PMPI_Group_translate_ranks(local_group,
                                 1,
                                 &target_rank,
                                 global_group,
                                 &global_rank);

      vftr_register_onesided_request(send, origin_count, origin_datatype,
                                     global_rank, MPI_COMM_WORLD, *request, tstart);
      long long t2end = vftr_get_runtime_usec();

      vftr_mpi_overhead_usec += t2end - t2start;

      return retVal;
   }
}

#endif
