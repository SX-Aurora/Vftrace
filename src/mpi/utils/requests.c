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
#include <stdbool.h>

#include "requests.h"
#include "p2p_requests.h"
#include "onesided_requests.h"
#include "collective_requests.h"
#include "stacks.h"
#include "threads.h"
#include "threadstacks.h"


int vftr_open_request_list_length = 0;
vftr_request_t *vftr_open_request_list = NULL;

// create new request to be stored
vftr_request_t* vftr_register_request(message_direction dir, int nmsg, int *count,
                                      MPI_Datatype *type, int tag,
                                      MPI_Comm comm, MPI_Request request,
                                      int n_tmp_ptr, void **tmp_ptrs,
                                      long long tstart) {

   // search for the first invalidated request
   int invalid_request_id = -1;
   bool invalid_request = false;
   while (invalid_request == false &&
          (invalid_request_id+1)<vftr_open_request_list_length) {
      invalid_request_id++;
      invalid_request = (!vftr_open_request_list[invalid_request_id].valid);
   }

   // if no free spot was found reallocate
   if (!invalid_request) {
      invalid_request_id = vftr_open_request_list_length;
      vftr_open_request_list_length++;
      vftr_open_request_list = (vftr_request_t*) realloc(vftr_open_request_list,
                               vftr_open_request_list_length*sizeof(vftr_request_t));
   }

   // fill data into request
   vftr_request_t *new_request = vftr_open_request_list+invalid_request_id;
   new_request->valid     = true;
   new_request->request   = request;
   new_request->comm      = comm;
   new_request->nmsg      = nmsg;
   new_request->dir       = dir;
   new_request->count     = (int*) malloc(sizeof(int)*nmsg);
   for (int i=0; i<nmsg; i++) {
      new_request->count[i] = count[i];
   }
   new_request->type      = (MPI_Datatype*) malloc(sizeof(MPI_Datatype)*nmsg);
   for (int i=0; i<nmsg; i++) {
      new_request->type[i] = type[i];
   }
   new_request->type_idx  = (int*) malloc(sizeof(int)*nmsg);
   new_request->type_size = (int*) malloc(sizeof(int)*nmsg);
   for (int i=0; i<nmsg; i++) {
      // Determine type index in vftrace type list
      // and its size in bytes
      int type_idx = vftr_get_mpitype_idx(type[i]);
      int type_size;
      if (type[i] != MPI_DATATYPE_NULL) {
         PMPI_Type_size(type[i], &type_size);
      } else {
         type_size = 0;
      }
      new_request->type_idx[i]  = type_idx;
      new_request->type_size[i] = type_size;
   }
   new_request->tstart    = tstart;
   new_request->marked_for_deallocation = false;

   // rank and tag
   // Due to differences in how they are filled within P2P, collective, onesided
   // communications, only memory space is provided and is to be filled later.
   new_request->tag = tag;
   new_request->rank = (int*) malloc(sizeof(int)*nmsg);
   // Get the thread that called the function
   thread_t *my_thread = vftr_get_my_thread(&(vftrace.process.threadtree));
   threadstack_t *my_threadstack = vftr_get_my_threadstack(my_thread);

   new_request->callingstackID = my_threadstack->stackID;

   // store temporary pointers used for mpi-communication
   new_request->n_tmp_ptr = n_tmp_ptr;
   new_request->tmp_ptrs = tmp_ptrs;

   return new_request;
}

// clear the requests and log the messaging
void vftr_clear_completed_requests() {
   int mpi_isinit;
   PMPI_Initialized(&mpi_isinit);
   if (mpi_isinit) {
      int mpi_isfinal;
      PMPI_Finalized(&mpi_isfinal);
      if (!mpi_isfinal) {
         for (int ireq=0; ireq<vftr_open_request_list_length; ireq++) {
            vftr_request_t *current_request = vftr_open_request_list+ireq;
            // only attempt to clear it if it is valid
            if (current_request->valid &&
                (!current_request->persistent ||
                 (current_request->persistent && current_request->active))) {
               switch (current_request->request_kind) {
                  case p2p:
                     vftr_clear_completed_p2p_request(current_request);
                     break;
                  case onesided:
                     vftr_clear_completed_onesided_request(current_request);
                     break;
                  case collective:
                     vftr_clear_completed_collective_request(current_request);
                     break;
                  default:
                     // TODO: Add error handling
                     ;
               }
            }
         }
      }
   }
}

void vftr_activate_pers_request(MPI_Request request, long long tstart) {
   // search for request in open request list
   vftr_request_t *matching_request = vftr_search_request(request);
   if (matching_request != NULL) {
printf("activating request to peer %d\n", matching_request->rank[0]);
      matching_request->active = true;
      matching_request->tstart = tstart;
   }
}

// remove a request
void vftr_remove_request(vftr_request_t *request) {
   if (request->valid) {
      request->valid = false;
      request->request = MPI_REQUEST_NULL;
      free(request->count);
      request->count = NULL;
      free(request->type);
      request->type = NULL;
      free(request->type_idx);
      request->type_idx = NULL;
      free(request->type_size);
      request->type_size = NULL;
      free(request->rank);
      request->rank = NULL;

      // free temporary pointers
      for (int ireq=0; ireq<request->n_tmp_ptr; ireq++) {
         free(*(request->tmp_ptrs+ireq));
         *(request->tmp_ptrs+ireq) = NULL;
      }
      free(request->tmp_ptrs);
      request->tmp_ptrs = NULL;
   }
}

// deallocate entire request list
void vftr_free_request_list() {
   if (vftr_open_request_list_length > 0) {
      vftr_open_request_list_length = 0;
      free(vftr_open_request_list);
      vftr_open_request_list = NULL;
   }
}

// find a specific request in the request list.
vftr_request_t *vftr_search_request(MPI_Request request) {
   // go through the complete list and check the request
   vftr_request_t *matching_request = NULL;
   int request_id = 0;
   while (request_id < vftr_open_request_list_length && matching_request == NULL) {
      vftr_request_t *tmprequest = vftr_open_request_list+request_id;
      if (tmprequest->request == request) {
         matching_request = tmprequest;
      } else {
         request_id++;
      }
   }
   return matching_request;
}
