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

#include "vftr_timer.h"
#include "vftr_regions.h"
#include "vftr_environment.h"
#include "vftr_sync_messages.h"
#include "vftr_mpi_utils.h"
#include "vftr_mpi_buf_addr_const.h"

int vftr_MPI_Bcast(void *buffer, int count, MPI_Datatype datatype,
                   int root, MPI_Comm comm) {

   // disable profiling based on the Pcontrol level
   if (vftr_no_mpi_logging()) {
      return PMPI_Bcast(buffer, count, datatype, root, comm);
   } else {
      // Estimate synchronization time
      if (vftr_environment->mpi_show_sync_time->value) {
         vftr_internal_region_begin("MPI_Bcast_sync");
         PMPI_Barrier(comm);
         vftr_internal_region_end("MPI_Bcast_sync");
      }

      long long tstart = vftr_get_runtime_usec();
      int retVal = PMPI_Bcast(buffer, count, datatype, root, comm);
      long long tend = vftr_get_runtime_usec();

      long long t2start = tend;
      if (vftr_env_do_sampling()) {
         // determine if inter or intra communicator
         int isintercom;
         PMPI_Comm_test_inter(comm, &isintercom);
         if (isintercom) {
            // in intercommunicators the behaviour is more complicated
            // There are two groups A and B
            // In group A the root process is located.
            if (root == MPI_ROOT) {
               // The root process get the special process wildcard MPI_ROOT
               // get the size of group B
               int size;
               PMPI_Comm_remote_size(comm, &size);
               for (int i=0; i<size; i++) {
                  // translate the i-th rank in group B to the global rank
                  int global_peer_rank = vftr_remote2global_rank(comm, i);
                  // store message info with MPI_COMM_WORLD as communicator
                  // to prevent additional (and thus faulty rank translation)
                  vftr_store_sync_message_info(send, count, datatype,
                                               global_peer_rank, -1, MPI_COMM_WORLD,
                                               tstart, tend);
               }
            } else if (root == MPI_PROC_NULL) {
               // All other processes from group A pass wildcard MPI_PROC NULL
               // They do not participate in intercommunicator gather
               ;
            } else {
               // All other processes must be located in group B
               // root is the rank-id in group A Therefore no problems with 
               // rank translation should arise
               vftr_store_sync_message_info(recv, count, datatype,
                                            root, -1, comm, tstart, tend);
            }
         } else {
            // in intracommunicators the expected behaviour is to
            // gather from root to all other processes in the communicator
            int rank;
            PMPI_Comm_rank(comm, &rank);
            if (rank == root) {
               int size;
               PMPI_Comm_size(comm, &size);
               for (int i=0; i<size; i++) {
                  vftr_store_sync_message_info(send, count, datatype, i, -1,
                                               comm, tstart, tend);
               }
            } else {
               vftr_store_sync_message_info(recv, count, datatype, root, -1,
                                            comm, tstart, tend);
            }
         }
      }
      long long t2end = vftr_get_runtime_usec();

      vftr_mpi_overhead_usec += t2end - t2start;

      return retVal;
   }
}

#endif
