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

#include <stdlib.h>

#ifdef _MPI
#include "vftr_mpi_utils.h"
#include "vftr_environment.h"
#include "vftr_filewrite.h"
#include "vftr_timer.h"
#include "vftr_pause.h"

// store open requests as doubly linked list
typedef struct vftr_request_list_type {
   struct vftr_request_list_type *prev, *next;
   MPI_Request request;
   MPI_Comm comm;
   int dir;
   int count;
   MPI_Datatype type;
   int type_idx;
   int type_size;
   int rank;
   int tag;
   long long tstart;
} vftr_request_list_t;

vftr_request_list_t *vftr_open_request_list = NULL;

// store message info for asynchronous mpi-communication
void vftr_register_request(vftr_direction dir, int count, MPI_Datatype type, 
                           int peer_rank, int tag, MPI_Comm comm,
                           MPI_Request request, long long tstart) {

   // only continue if sampling and mpi_loggin is enabled
   bool mpi_log = vftr_environment->mpi_log->value;
   if (vftr_off() || !mpi_log || vftr_paused) return;

   // immediately return if peer is MPI_PROC_NULL as this is a dummy rank
   // with no effect on communication at all
   if (peer_rank == MPI_PROC_NULL) {return;}

   // if the communicator is 
   int type_idx = vftr_get_mpitype_idx(type);
   int type_size;
   if (type != MPI_DATATYPE_NULL) {
      PMPI_Type_size(type, &type_size);
   } else {
      type_size = 0;
   }
   
   // create new request list entry
   vftr_request_list_t *new_open_request = (vftr_request_list_t*) 
      malloc(sizeof(vftr_request_list_t));
   new_open_request->request   = request;
   new_open_request->comm      = comm;
   new_open_request->dir       = dir;
   new_open_request->count     = count;
   new_open_request->type      = type;
   new_open_request->type_idx  = type_idx;
   new_open_request->type_size = type_size;
   // rank and tag are stored 
   // If they are wildcards (like MPI_ANY_SOURCE, MPI_ANY_TAG) they will be overwritten
   // as soon as the communication is complieted with the real values
   // extracted from the completed communication status 
   new_open_request->tag = tag;
   if (peer_rank != MPI_ANY_SOURCE) {
      // check if the communicator is an intercommunicator
      int isintercom;
      PMPI_Comm_test_inter(comm, &isintercom);
      if (isintercom) {
         new_open_request->rank = vftr_remote2global_rank(comm, peer_rank);
      } else {
         new_open_request->rank = vftr_local2global_rank(comm, peer_rank);
      }
   } else {
      new_open_request->rank = MPI_ANY_SOURCE;
   }
   new_open_request->tstart    = tstart;

   // prepend message info to list for open communication
   if (vftr_open_request_list == NULL) {
      // list is empty. this is the first element
      vftr_open_request_list = new_open_request;
      vftr_open_request_list->prev = NULL;
      vftr_open_request_list->next = NULL;
   } else {
      // list contains entries
      new_open_request->next = vftr_open_request_list;
      new_open_request->prev = NULL;
      vftr_open_request_list->prev = new_open_request;
      vftr_open_request_list = new_open_request;
   }

   // increase list for vftr_request_list
   return;
}
                           
// test the entire list of open request for completed communication
void vftr_clear_completed_request() {
   // only continue if sampling and mpi_loggin is enabled
   bool mpi_log = vftr_environment->mpi_log->value;
   if (vftr_off() || !mpi_log) return;

   // go through the complete list and check the request
   vftr_request_list_t *current_request = vftr_open_request_list;
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
         if (current_request->rank == MPI_ANY_SOURCE) {
            current_request->rank = tmpStatus.MPI_SOURCE;
            // check if the communicator is an intercommunicator
            int isintercom;
            PMPI_Comm_test_inter(current_request->comm, &isintercom);
            if (isintercom) {
               current_request->rank = 
                  vftr_remote2global_rank(current_request->comm, current_request->rank);
            } else {
               current_request->rank = 
                  vftr_local2global_rank(current_request->comm, current_request->rank);
            }
         }
         // Get the actual amount of transferred data if it is a receive operation
         if (current_request->dir == recv) {
            int tmpcount;
            PMPI_Get_count(&tmpStatus, current_request->type, &tmpcount);
            if (tmpcount != MPI_UNDEFINED) {
               current_request->count = tmpcount;
            }
         }
         // store the completed communication info to the outfile
         vftr_store_message_info(current_request->dir,
                                 current_request->count,
                                 current_request->type_idx,
                                 current_request->type_size,
                                 current_request->rank,
                                 current_request->tag,
                                 current_request->tstart,
                                 tend);
         // remove the request from the open request list
         // create a temporary pointer to the current element to be used for deallocation
         vftr_request_list_t *tmp_current_request = current_request;
         // connect the previous element of the list with the next one
         if (current_request->prev == NULL) {
            if (current_request->next == NULL) {
               // only entry in the list
               vftr_open_request_list = NULL;
            } else {
               // first element in the list
               vftr_open_request_list = current_request->next;
               current_request->next->prev = NULL;
            }
         } else {
            if (current_request->next == NULL) {
               // last element in the list
               current_request->prev->next = NULL;
            } else {
               // somewhere in the middle of the list
               current_request->prev->next = current_request->next;
               current_request->next->prev = current_request->prev;
            }
         }
         // advance in list
         current_request = current_request->next;
         // deallocate obsolete entry
         free(tmp_current_request);
         tmp_current_request = NULL;
      } else {
         // advance in list
         current_request = current_request->next;
      }
   } // end of while loop
   
   return;
}

#endif
