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

int vftr_MPI_Recv(void *buf, int count, MPI_Datatype datatype,
                  int source, int tag, MPI_Comm comm, MPI_Status *status) {
   MPI_Status tmpstatus;
   long long tstart = vftr_get_runtime_usec();
   int retVal = PMPI_Recv(buf, count, datatype, source, tag, comm, &tmpstatus);
   long long tend = vftr_get_runtime_usec();

   long long t2start = tend;
   vftr_store_sync_message_info(recv, count, datatype, tmpstatus.MPI_SOURCE,
      tmpstatus.MPI_TAG, comm, tstart, tend);

   // handle the special case of MPI_STATUS_IGNORE
   if (status != MPI_STATUS_IGNORE) {
      status->MPI_SOURCE = tmpstatus.MPI_SOURCE;
      status->MPI_TAG = tmpstatus.MPI_TAG;
      status->MPI_ERROR = tmpstatus.MPI_ERROR;
   }

   thread_t *my_thread = vftr_get_my_thread(&(vftrace.process.threadtree));
   threadstack_t *my_threadstack = vftr_get_my_threadstack(my_thread);
   stack_t *my_stack = vftrace.process.stacktree.stacks+my_threadstack->stackID;
   profile_t *my_profile = vftr_get_my_profile(my_stack, my_thread);
   long long t2end = vftr_get_runtime_usec();

   vftr_accumulate_mpi_overheadprofiling(&(my_profile->overheadProf), t2end-t2start);

   return retVal;
}
