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

#include "vftr_environment.h"
#include "vftr_timer.h"
#include "vftr_sync_messages.h"
#include "vftr_mpi_utils.h"

int vftr_MPI_Sendrecv(const void *sendbuf, int sendcount,
                      MPI_Datatype sendtype, int dest, int sendtag,
                      void *recvbuf, int recvcount, MPI_Datatype recvtype,
                      int source, int recvtag, MPI_Comm comm,
                      MPI_Status *status) {

   // disable profiling based on the Pcontrol level
   if (vftr_no_mpi_logging()) {
      return PMPI_Sendrecv(sendbuf, sendcount, sendtype, dest, sendtag,
                           recvbuf, recvcount, recvtype, source, recvtag,
                           comm, status);
   } else {
      MPI_Status tmpstatus;
      long long tstart = vftr_get_runtime_usec();
      int retVal = PMPI_Sendrecv(sendbuf, sendcount, sendtype, dest, sendtag,
                                 recvbuf, recvcount, recvtype, source, recvtag,
                                 comm, &tmpstatus);
      long long tend = vftr_get_runtime_usec();
  
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
      long long t2end = vftr_get_runtime_usec();

      vftr_mpi_overhead_usec += t2end - t2start;
  
      return retVal;
   }
}

#endif
