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

#include "vftr_environment.h"
#include "vftr_requests.h"
#include "vftr_timer.h"
#include "vftr_filewrite.h"

vftr_request_t *vftr_open_collective_request_list = NULL;

void vftr_register_collective_request(vftr_direction dir, int nmsg, int *count,
                                      MPI_Datatype *type, int *peer_rank,
                                      MPI_Comm comm, MPI_Request request,
                                      int n_tmp_ptr, void **tmp_ptrs,
                                      long long tstart) {

   vftr_request_t *new_request = vftr_new_request(dir, nmsg, count, type, -1, comm, request, n_tmp_ptr, tmp_ptrs, tstart);
   int isintercom;
   PMPI_Comm_test_inter(comm, &isintercom);
   if (isintercom) {
      for (int i=0; i<nmsg; i++) {
         new_request->rank[i] = vftr_remote2global_rank(comm, peer_rank[i]);
      }
   } else {
      for (int i=0; i<nmsg; i++) {
         new_request->rank[i] = vftr_local2global_rank(comm, peer_rank[i]);
      }
   }

   vftr_request_prepend(&vftr_open_collective_request_list, new_request);
}

void vftr_clear_completed_collective_requests() {
   // go through the complete list and check the request
   vftr_request_t *current_request = vftr_open_collective_request_list;
   while (current_request != NULL) {
      // Test if the current request is still ongoing or completed
      // Note: MPI_Test is a destructive test. It will destroy the request
      //       Thus, running with and without vftr can lead to different program executions
      //       MPI_Request_get_status is a non destructive status check
      int flag;
      PMPI_Request_get_status(current_request->request,
                              &flag,
                              MPI_STATUS_IGNORE);

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
         if (vftr_environment.do_sampling->value) {
            for (int i=0; i<current_request->nmsg; i++) {
               vftr_store_message_info(current_request->dir,
                                       current_request->count[i],
                                       current_request->type_idx[i],
                                       current_request->type_size[i],
                                       current_request->rank[i],
                                       current_request->tag,
                                       current_request->tstart,
                                       tend,
                                       current_request->callingstackID);
            }
         }

         // Take the request out of the list and close the gap
         vftr_remove_request(&vftr_open_collective_request_list, current_request);

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

int vftr_number_of_open_collective_requests() {
   int nrequests = 0;
   // go through the complete list and check the request
   vftr_request_t *current_request = vftr_open_collective_request_list;
   while (current_request != NULL) {
      nrequests++;
      current_request = current_request->next;
   }
   return nrequests;
}


#endif
