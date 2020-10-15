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

#ifdef _MPI
#include <stdbool.h>

#include "vftr_mpi_utils.h"

// store open requests as doubly linked list
typedef struct vftr_request_type {
   struct vftr_request_type *prev, *next;
   MPI_Request request;
   bool marked_for_deallocation;
   MPI_Comm comm;
   int nmsg;
   int dir;
   int *count;
   MPI_Datatype *type;
   int *type_idx;
   int *type_size;
   int *rank;
   int tag;
   long long tstart;
} vftr_request_t;

// create new request to be stored
vftr_request_t* vftr_new_request(vftr_direction dir, int nmsg, int *count,
                                 MPI_Datatype *type, int tag,
                                 MPI_Comm comm, MPI_Request request,
                                 long long tstart);

// free a request
void vftr_free_request(vftr_request_t **request_ptr);

// prepend request to open_request_list
void vftr_request_prepend(vftr_request_t **open_request_list,
                          vftr_request_t *new_request);

// remove request from open_request_list
void vftr_remove_request(vftr_request_t **open_request_list,
                         vftr_request_t *request);

// find a specific request in the request list.
vftr_request_t *vftr_search_request(vftr_request_t *open_request_list,
                                    MPI_Request request);

#endif
#endif
