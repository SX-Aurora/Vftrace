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
#include "vftr_mpi_pcontrol.h"
#include "vftr_mpi_buf_addr_const.h"

int vftr_MPI_Allgatherv(const void *sendbuf, int sendcount,
                        MPI_Datatype sendtype, void *recvbuf,
                        const int *recvcounts, const int *displs,
                        MPI_Datatype recvtype, MPI_Comm comm) {

   // disable profiling based on the Pcontrol level
   if (vftrace_Pcontrol_level == 0) {
      return PMPI_Allgatherv(sendbuf, sendcount, sendtype, recvbuf,
                             recvcounts, displs, recvtype, comm);
   } else {
      // Estimate synchronization time
      if (vftr_environment->mpi_show_sync_time->value) {
         vftr_internal_region_begin("mpi_allgatherv_sync");
         PMPI_Barrier(comm);
         vftr_internal_region_end("mpi_allgatherv_sync");
      }
      
      long long tstart = vftr_get_runtime_usec();
      int retVal = PMPI_Allgatherv(sendbuf, sendcount, sendtype, recvbuf,
                                   recvcounts, displs, recvtype, comm);
      long long tend = vftr_get_runtime_usec();
  
      // determine if inter or intra communicator
      int isintercom;
      PMPI_Comm_test_inter(comm, &isintercom);
      if (isintercom) {
         // Every process of group A sends sendcount data to and
         // receives recvcounts[i] data from the i-th process in group B
         int size;
         PMPI_Comm_remote_size(comm, &size);
         for (int i=0; i<size; i++) {
            // translate the i-th rank in the remote group to the global rank
            int global_peer_rank = vftr_remote2global_rank(comm, i);
            // Store message info with MPI_COMM_WORLD as communicator
            // to prevent additional (and thus faulty rank translation)
            vftr_store_sync_message_info(send, sendcount, sendtype,
                                         global_peer_rank, -1, MPI_COMM_WORLD,
                                         tstart, tend);
            vftr_store_sync_message_info(recv, recvcounts[i], recvtype,
                                         global_peer_rank, -1, MPI_COMM_WORLD,
                                         tstart, tend);
         }
      } else {
         int size;
         PMPI_Comm_size(comm, &size);
         // if sendbuf is special address MPI_IN_PLACE
         // sendcount and sendtype are ignored.
         // Use recvcount and recvtype for statistics
         if (vftr_is_C_MPI_IN_PLACE(sendbuf)) {
            int rank;
            PMPI_Comm_rank(comm, &rank);

            sendcount = recvcounts[rank];
            sendtype = recvtype;
            // For the in-place option no self communication is executed
            for (int i=0; i<rank; i++) {
                vftr_store_sync_message_info(send, sendcount, sendtype, i, -1,
                                             comm, tstart, tend);
                vftr_store_sync_message_info(recv, recvcounts[i], recvtype, i, -1,
                                             comm, tstart, tend);
             }
            for (int i=rank+1; i<size; i++) {
                vftr_store_sync_message_info(send, sendcount, sendtype, i, -1,
                                             comm, tstart, tend);
                vftr_store_sync_message_info(recv, recvcounts[i], recvtype, i, -1,
                                             comm, tstart, tend);
             }

         } else {
            for (int i=0; i<size; i++) {
               vftr_store_sync_message_info(send, sendcount, sendtype, i, -1,
                                            comm, tstart, tend);
               vftr_store_sync_message_info(recv, recvcounts[i], recvtype, i, -1,
                                            comm, tstart, tend);
            }
         }
      }
  
      return retVal;
   }
}

#endif
