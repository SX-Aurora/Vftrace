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
#include "thread_types.h"
#include "threads.h"
#include "threadstack_types.h"
#include "threadstacks.h"
#include "stack_types.h"
#include "stacks.h"
#include "profiling_types.h"
#include "profiling.h"
#include "mpiprofiling.h"
#include "timer.h"
#include "sync_messages.h"

int vftr_MPI_Sendrecv(const void *sendbuf, int sendcount,
                      MPI_Datatype sendtype, int dest, int sendtag,
                      void *recvbuf, int recvcount, MPI_Datatype recvtype,
                      int source, int recvtag, MPI_Comm comm,
                      MPI_Status *status) {
   MPI_Status tmpstatus;
   long long tstart = vftr_get_runtime_nsec();
   int retVal = PMPI_Sendrecv(sendbuf, sendcount, sendtype, dest, sendtag,
                              recvbuf, recvcount, recvtype, source, recvtag,
                              comm, &tmpstatus);
   long long tend = vftr_get_runtime_nsec();

   SELF_PROFILE_START_FUNCTION;
   long long t2start = tend;
   int rank;
   PMPI_Comm_rank(comm, &rank);
   vftr_store_sync_message_info(send, sendcount, sendtype, dest,
                                sendtag, comm, tstart, tend);
   vftr_store_sync_message_info(recv, recvcount, recvtype,
                                tmpstatus.MPI_SOURCE, tmpstatus.MPI_TAG,
                                comm, tstart, tend);

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
   long long t2end = vftr_get_runtime_nsec();

   vftr_accumulate_mpiprofiling_overhead(&(my_profile->mpiprof), t2end-t2start);

   SELF_PROFILE_END_FUNCTION;
   return retVal;
}
