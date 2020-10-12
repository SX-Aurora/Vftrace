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

#ifndef VFTR_P2P_REQUESTS_H
#define VFTR_P2P_REQUESTS_H

#ifdef _MPI
#include "vftr_requests.h"

void vftr_register_P2P_request(vftr_direction dir, int count,
                               MPI_Datatype type, int peer_rank, int tag,
                               MPI_Comm comm, MPI_Request request,
                               long long tstart);

void vftr_clear_completed_P2P_requests();

bool vftr_mark_p2p_request_for_deallocation(MPI_Request request);

void vftr_deallocate_marked_p2p_requests();

int vftr_number_of_open_p2p_requests();

#endif
#endif
