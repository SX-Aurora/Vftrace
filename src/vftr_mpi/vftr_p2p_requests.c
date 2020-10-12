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

#include <stdlib.h>

#include "vftr_requests.h"
#include "vftr_timer.h"
#include "vftr_filewrite.h"

vftr_request_t *vftr_open_p2p_request_list = NULL;

void vftr_register_P2P_request(vftr_direction dir, int count,
                               MPI_Datatype type, int peer_rank, int tag,
                               MPI_Comm comm, MPI_Request request,
                               long long tstart) {

   // immediately return if peer is MPI_PROC_NULL as this is a dummy rank
   // with no effect on communication at all
   if (peer_rank == MPI_PROC_NULL) {return;}

   vftr_request_t *new_request = vftr_new_request(dir, 1, &count, &type, tag, comm, request, tstart);
   new_request->rank[0] = peer_rank;

   vftr_request_prepend(&vftr_open_p2p_request_list, new_request);
}

void vftr_clear_completed_P2P_requests() {
   // go through the complete list and check the request
   vftr_request_t *current_request = vftr_open_p2p_request_list;
   while (current_request != NULL) {
      // Test if the current request is still ongoing or completed
      // Note: MPI_Test is a destructive test. It will destroy the request
      //       Thus, running with and without vftr can lead to different program executions
      //       MPI_Request_get_status is a non destructive status check
      MPI_Status tmpStatus;
      int flag;
      PMPI_Request_get_status(current_request->request,
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
         if (current_request->tag == MPI_ANY_TAG) {
            current_request->tag = tmpStatus.MPI_TAG;
         }
         if (current_request->rank[0] == MPI_ANY_SOURCE) {
            current_request->rank[0] = tmpStatus.MPI_SOURCE;
            current_request->rank[0] = vftr_local2global_rank(current_request->comm,
                                                              current_request->rank[0]);
         }

         // Get the actual amount of transferred data if it is a receive operation
         // only if it was a point2point communication
         if (current_request->dir == recv) {
            int tmpcount;
            PMPI_Get_count(&tmpStatus, current_request->type[0], &tmpcount);
            if (tmpcount != MPI_UNDEFINED) {
               current_request->count[0] = tmpcount;
            }
         }
         // store the completed communication info to the outfile
         vftr_store_message_info(current_request->dir,
                                 current_request->count[0],
                                 current_request->type_idx[0],
                                 current_request->type_size[0],
                                 current_request->rank[0],
                                 current_request->tag,
                                 current_request->tstart,
                                 tend);

         // Take the request out of the list and close the gap
         vftr_remove_request(&vftr_open_p2p_request_list, current_request);

         // create a temporary pointer to the current element to be used for deallocation
         vftr_request_t * tmp_current_request = current_request;
         // advance in list
         current_request = current_request->next;
         vftr_free_request(&tmp_current_request);
      } else {
         // advance in list
         current_request = current_request->next;
      }
   } // end of while loop
}

bool vftr_mark_p2p_request_for_deallocation(MPI_Request request) {
   // check if the request is in the list of open p2p requests
   vftr_request_t *matching_p2p_request = vftr_search_request(vftr_open_p2p_request_list, request);
   if (matching_p2p_request != NULL) {
      matching_p2p_request->marked_for_deallocation = true;
      return true;
   } else {
      return false;
   }
}

void vftr_deallocate_marked_p2p_requests() {
   // go through the complete list and check the request
   vftr_request_t *current_request = vftr_open_p2p_request_list;
   while (current_request != NULL) {
      if (current_request->marked_for_deallocation) {
         // Take the request out of the list and close the gap
         vftr_remove_request(&vftr_open_p2p_request_list, current_request);

         // create a temporary pointer to the current element to be used for deallocation
         vftr_request_t * tmp_current_request = current_request;
         // advance in list
         current_request = current_request->next;
         vftr_free_request(&tmp_current_request);
      } else {
         // advance in list
         current_request = current_request->next;
      }
   }
}

int vftr_number_of_open_p2p_requests() {
   int nrequests = 0;
   // go through the complete list and check the request
   vftr_request_t *current_request = vftr_open_p2p_request_list;
   while (current_request != NULL) {
      nrequests++;
      current_request = current_request->next;
   }
   return nrequests;
}


#endif
