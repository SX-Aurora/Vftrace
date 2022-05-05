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

#include <stdlib.h>

#include "vftrace_state.h"
#include "rank_translate.h"
#include "requests.h"
#include "timer.h"
#include "mpi_logging.h"

void vftr_register_onesided_request(message_direction dir, int count,
                                    MPI_Datatype type, int peer_rank,
                                    MPI_Comm comm, MPI_Request request,
                                    long long tstart) {

   // immediately return if peer is MPI_PROC_NULL as this is a dummy rank
   // with no effect on communication at all
   if (peer_rank == MPI_PROC_NULL) {return;}

   vftr_request_t *new_request = vftr_register_request(dir, 1, &count, &type, -1, comm, request, 0, NULL, tstart);
   new_request->rank[0] = vftr_local2global_rank(comm, peer_rank);
   new_request->request_kind = onesided;
   new_request->persistent = false;
}

void vftr_clear_completed_onesided_request(vftr_request_t *request) {
   // Test if the current request is still ongoing or completed
   // Note: MPI_Test is a destructive test. It will destroy the request
   //       Thus, running with and without vftr can lead to different program executions
   //       MPI_Request_get_status is a non destructive status check
   MPI_Status tmpStatus;
   int flag;
   PMPI_Request_get_status(request->request,
                           &flag,
                           &tmpStatus);

   // if the requested communication is finished write the communication out
   // and remove the request from the request list
   if (flag) {
      // record the time when the communication is finished
      // (might be to late, but thats as accurate as we can make it
      //  without violating the MPI-Standard)
      // Therefore: measures asynchronous communication with vftrace always
      //            yields to small bandwidth.
      long long tend = vftr_get_runtime_usec ();

      // Every rank should already be translated to the global rank
      // by the register routine
      // store the completed communication info to the outfile
      if (vftrace.environment.do_sampling.value.bool_val) {
         vftr_store_message_info(request->dir,
                                 request->count[0],
                                 request->type_idx[0],
                                 request->type_size[0],
                                 request->rank[0],
                                 request->tag,
                                 request->tstart,
                                 tend,
                                 request->callingstackID);
      }

      // Take the request out of the list
      vftr_remove_request(request);
   }
}
