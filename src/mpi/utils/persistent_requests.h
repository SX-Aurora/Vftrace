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

#ifndef VFTR_PERSISTENT_REQUESTS_H
#define VFTR_PERSISTENT_REQUESTS_H

#include "requests.h"

extern vftr_request_t *vftr_open_persistent_request_list;

void vftr_register_persistent_request(vftr_direction dir, int count,
                                      MPI_Datatype type, int peer_rank, int tag,
                                      MPI_Comm comm, MPI_Request request);

void vftr_activate_persistent_request(MPI_Request request, long long tstart);

void vftr_deactivate_completed_persistent_requests();

vftr_request_t *vftr_search_persistent_request(MPI_Request request);

int vftr_number_of_open_persistent_requests();

#endif
