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
#include "vftr_mpi_utils.h"
#include "vftr_environment.h"
#include "vftr_filewrite.h"
#include "vftr_stacks.h"
#include "vftr_pause.h"

// store message info for synchronous mpi-communication
void vftr_store_sync_message_info(vftr_direction dir, int count, MPI_Datatype type,
                                  int peer_rank, int tag, MPI_Comm comm, 
                                  long long tstart, long long tend) {

   // only continue if sampling and mpi_loggin is enabled
   bool mpi_log = vftr_environment.mpi_log->value;
   if (vftr_off() || !mpi_log || vftr_paused) return;

   // immediately return if peer is MPI_PROC_NULL as this is a dummy rank
   // with no effect on communication at all
   if (peer_rank == MPI_PROC_NULL) {return;}

   // translate rank to global in case the communicator is not global
   int rank = peer_rank;
   if (comm != MPI_COMM_WORLD) {
     // check if the communicator is an intercommunicator
     int isintercom;
     PMPI_Comm_test_inter(comm, &isintercom);
     if (isintercom) {
        rank = vftr_remote2global_rank(comm, peer_rank);
     } else {
        rank = vftr_local2global_rank(comm, peer_rank);
     }
   }

   int type_idx = vftr_get_mpitype_idx(type);
   int type_size;
   if (type != MPI_DATATYPE_NULL) {
      PMPI_Type_size(type, &type_size);
   } else {
      type_size = 0;
   }

   // accumulate information for later use in the log file statistics
   vftr_log_message_info(dir, count, type_idx, type_size, rank, tag, tstart, tend);

   // store message in vfd-file
   if (vftr_environment.do_sampling->value) {
      vftr_store_message_info(dir, count, type_idx, type_size, rank, tag, tstart, tend, vftr_fstack->id);
   }

   return;
}

#endif
