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

vftr_request_t *vftr_open_persistent_request_list = NULL;

void vftr_register_persistent_request(vftr_direction dir, int count,
                                      MPI_Datatype type, int peer_rank, int tag,
                                      MPI_Comm comm, MPI_Request request) {

   // immediately return if peer is MPI_PROC_NULL as this is a dummy rank
   // with no effect on communication at all
   if (peer_rank == MPI_PROC_NULL) {return;}

   vftr_request_t *new_request = vftr_new_request(dir, 1, &count, &type, tag, comm, request, 0ll);
   new_request->rank[0] = peer_rank;

   vftr_request_prepend(&vftr_open_persistent_request_list, new_request);
}

#endif
