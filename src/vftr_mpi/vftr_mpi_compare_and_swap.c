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
#include "vftr_sync_messages.h"
#include "vftr_mpi_pcontrol.h"

int vftr_MPI_Compare_and_swap(const void *origin_addr, const void *compare_addr,
                              void *result_addr, MPI_Datatype datatype,
                              int target_rank, MPI_Aint target_disp, MPI_Win win) {

   // disable profiling based on the Pcontrol level
   if (vftrace_Pcontrol_level == 0) {
      return PMPI_Compare_and_swap(origin_addr, compare_addr, result_addr,
                                   datatype, target_rank, target_disp, win);
   } else {
      long long tstart = vftr_get_runtime_usec();
      int retVal = PMPI_Compare_and_swap(origin_addr, compare_addr, result_addr,
                                         datatype, target_rank, target_disp, win);
      long long tend = vftr_get_runtime_usec();

      long long t2start = tend;
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

      vftr_store_sync_message_info(recv, 1, datatype, global_rank,
                                   -1, MPI_COMM_WORLD, tstart, tend);
      vftr_store_sync_message_info(send, 1, datatype, global_rank,
                                   -1, MPI_COMM_WORLD, tstart, tend);
      long long t2end = vftr_get_runtime_usec();

      vftr_mpi_overhead_usec += t2end - t2start;

      return retVal;
   }
}

#endif
