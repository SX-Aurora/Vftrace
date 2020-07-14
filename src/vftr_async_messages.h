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

#ifndef VFTR_ASYNC_MESSAGES_H
#define VFTR_ASYNC_MESSAGES_H

#ifdef _MPI
#include "vftr_mpi_utils.h"

// add an open communication request to the list of all open requests
void vftr_register_request(vftr_direction dir, int count, MPI_Datatype type, 
                           int peer_rank, int tag, MPI_Comm comm,
                           MPI_Request request, long long tstart);

// test the entire list of open request for completed communication
void vftr_clear_completed_request();

#endif
#endif
