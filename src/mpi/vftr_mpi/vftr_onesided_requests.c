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

vftr_request_t *vftr_open_onesided_request_list = NULL;

void vftr_register_onesided_request(vftr_direction dir, int count,
                                    MPI_Datatype type, int peer_rank,
                                    MPI_Comm comm, MPI_Request request,
                                    long long tstart) {

   // immediately return if peer is MPI_PROC_NULL as this is a dummy rank
   // with no effect on communication at all
   if (peer_rank == MPI_PROC_NULL) {return;}

   vftr_request_t *new_request = vftr_new_request(dir, 1, &count, &type, -1, comm, request, 0, NULL, tstart);
   new_request->rank[0] = vftr_local2global_rank(comm, peer_rank);

   vftr_request_prepend(&vftr_open_onesided_request_list, new_request);
}

void vftr_clear_completed_onesided_requests() {
   // go through the complete list and check the request
   vftr_request_t *current_request = vftr_open_onesided_request_list;
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

         // Every rank should already be translated to the global rank
         // by the register routine
         // store the completed communication info to the outfile
         if (vftr_environment.do_sampling->value) {
            vftr_store_message_info(current_request->dir,
                                    current_request->count[0],
                                    current_request->type_idx[0],
                                    current_request->type_size[0],
                                    current_request->rank[0],
                                    current_request->tag,
                                    current_request->tstart,
                                    tend,
                                    current_request->callingstackID);
	 }

         // Take the request out of the list and close the gap
         vftr_remove_request(&vftr_open_onesided_request_list, current_request);

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

#endif
