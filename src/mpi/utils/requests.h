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

#ifndef VFTR_REQUESTS_H
#define VFTR_REQUESTS_H

#include <stdbool.h>

#include <mpi.h>

#include "mpi_util_types.h"

typedef enum vftr_request_kind_t {
   p2p,
   collective,
   onesided
} vftr_request_kind;

// open requests
typedef struct vftr_request_type {
   bool valid;
   bool persistent;
   bool active;
   bool marked_for_deallocation;
   MPI_Request request;
   vftr_request_kind request_kind;
   MPI_Comm comm;
   int nmsg;
   vftr_direction dir;
   int *count;
   MPI_Datatype *type;
   int *type_idx;
   int *type_size;
   int *rank;
   int tag;
   long long tstart;
   int callingstackID;
   int n_tmp_ptr;
   void **tmp_ptrs;
} vftr_request_t;

// create new request to be stored
vftr_request_t* vftr_register_request(vftr_direction dir, int nmsg, int *count,
                                      MPI_Datatype *type, int tag,
                                      MPI_Comm comm, MPI_Request request,
                                      int n_tmp_ptr, void **tmp_ptrs,
                                      long long tstart);

// clear the requests and log the messaging
void vftr_clear_completed_requests();

void vftr_activate_pers_request(MPI_Request request, long long tstart);

void vftr_remove_request(vftr_request_t *request);

// deallocate entire request list
void vftr_free_request_list();

// find a specific request in the request list.
vftr_request_t *vftr_search_request(MPI_Request request);

#endif
