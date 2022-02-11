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

#include "rank_translate.h"
#include "vftr_environment.h"
#include "requests.h"
#include "vftr_timer.h"
#include "vftr_filewrite.h"

void vftr_register_p2p_request(vftr_direction dir, int count,
                               MPI_Datatype type, int peer_rank, int tag,
                               MPI_Comm comm, MPI_Request request,
                               long long tstart) {

   // immediately return if peer is MPI_PROC_NULL as this is a dummy rank
   // with no effect on communication at all
   if (peer_rank == MPI_PROC_NULL) {return;}

   vftr_request_t *new_request = vftr_register_request(dir, 1, &count, &type, tag, comm, request, 0, NULL, tstart);
   new_request->rank[0] = peer_rank;
   new_request->request_kind = p2p;
   new_request->persistent = false;
}

void vftr_register_pers_p2p_request(vftr_direction dir, int count,
                                    MPI_Datatype type, int peer_rank, int tag,
                                    MPI_Comm comm, MPI_Request request) {
   // immediately return if peer is MPI_PROC_NULL as this is a dummy rank
   // with no effect on communication at all
   if (peer_rank == MPI_PROC_NULL) {return;}

   vftr_request_t *new_request = vftr_register_request(dir, 1, &count, &type, tag, comm, request, 0, NULL, 0);
   new_request->rank[0] = peer_rank;
   new_request->request_kind = p2p;
   new_request->persistent = true;
   new_request->active = false;
}

void vftr_clear_completed_p2p_request(vftr_request_t *request) {
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

      // extract rank and tag from the completed communication status
      // (if necessary) this is to avoid errors with wildcard usage
      if (request->tag == MPI_ANY_TAG) {
         request->tag = tmpStatus.MPI_TAG;
      }
      if (request->rank[0] == MPI_ANY_SOURCE) {
         request->rank[0] = tmpStatus.MPI_SOURCE;
         request->rank[0] = vftr_local2global_rank(request->comm,
                                                           request->rank[0]);
      }

      // Get the actual amount of transferred data if it is a receive operation
      // only if it was a point2point communication
      if (request->dir == recv) {
         int tmpcount;
         PMPI_Get_count(&tmpStatus, request->type[0], &tmpcount);
         if (tmpcount != MPI_UNDEFINED) {
            request->count[0] = tmpcount;
         }
      }
      // store the completed communication info to the outfile
      if (vftr_environment.do_sampling->value) {
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

      if (request->persistent) {
         request->active = false;
         if (request->marked_for_deallocation) {
            PMPI_Request_free(&(request->request));
         }
      } else {
         if (request->marked_for_deallocation) {
            PMPI_Request_free(&(request->request));
         }
         vftr_remove_request(request);
      }
   }
}
