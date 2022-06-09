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

#include "thread_types.h"
#include "threads.h"
#include "threadstack_types.h"
#include "threadstacks.h"
#include "stack_types.h"
#include "stacks.h"
#include "profiling_types.h"
#include "profiling.h"
#include "overheadprofiling.h"
#include "timer.h"
#include "sync_messages.h"

int vftr_MPI_Fetch_and_op(const void *origin_addr, void *result_addr,
                          MPI_Datatype datatype, int target_rank,
                          MPI_Aint target_disp, MPI_Op op, MPI_Win win) {
   long long tstart = vftr_get_runtime_usec();
   int retVal = PMPI_Fetch_and_op(origin_addr, result_addr, datatype,
                                  target_rank, target_disp, op, win);
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


   thread_t *my_thread = vftr_get_my_thread(&(vftrace.process.threadtree));
   threadstack_t *my_threadstack = vftr_get_my_threadstack(my_thread);
   stack_t *my_stack = vftrace.process.stacktree.stacks+my_threadstack->stackID;
   profile_t *my_profile = vftr_get_my_profile(my_stack, my_thread);
   long long t2end = vftr_get_runtime_usec();

   vftr_accumulate_mpi_overheadprofiling(&(my_profile->overheadProf), t2end-t2start);

   return retVal;
}
