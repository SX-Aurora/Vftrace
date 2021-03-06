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

int vftr_MPI_Alltoall(const void *sendbuf, int sendcount,
                      MPI_Datatype sendtype, void *recvbuf, int recvcount,
                      MPI_Datatype recvtype, MPI_Comm comm) {

   // Estimate synchronization time
   if (vftr_environment.mpi_show_sync_time->value) {
      vftr_internal_region_begin("MPI_Alltoall_sync");
      PMPI_Barrier(comm);
      vftr_internal_region_end("MPI_Alltoall_sync");
   }

   // disable profiling based on the Pcontrol level
   if (vftr_no_mpi_logging()) {
      return PMPI_Alltoall(sendbuf, sendcount, sendtype, recvbuf,
                           recvcount, recvtype, comm);
   } else {
      long long tstart = vftr_get_runtime_usec();
      int retVal = PMPI_Alltoall(sendbuf, sendcount, sendtype, recvbuf,
                                 recvcount, recvtype, comm);
      long long tend = vftr_get_runtime_usec();
  
      long long t2start = tend;
      // determine if inter or intra communicator
      int isintercom;
      PMPI_Comm_test_inter(comm, &isintercom);
      if (isintercom) {
         // Every process of group A sends sendcount sendtypes 
         // to and receives recvcount recvtypes from 
         // every process in group Band vice versa.
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
            vftr_store_sync_message_info(recv, recvcount, recvtype,
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
            sendcount = recvcount;
            sendtype = recvtype;
            // For the in-place option no self communication is executed
            int rank;
            PMPI_Comm_rank(comm, &rank);
            for (int i=0; i<rank; i++) {
               vftr_store_sync_message_info(send, sendcount, sendtype, i, 0,
                                            comm, tstart, tend);
               vftr_store_sync_message_info(recv, recvcount, recvtype, i, 0,
                                            comm, tstart, tend);
            }
            for (int i=rank+1; i<size; i++) {
               vftr_store_sync_message_info(send, sendcount, sendtype, i, 0,
                                            comm, tstart, tend);
               vftr_store_sync_message_info(recv, recvcount, recvtype, i, 0,
                                            comm, tstart, tend);
            }

         } else {
            for (int i=0; i<size; i++) {
               vftr_store_sync_message_info(send, sendcount, sendtype, i, 0,
                                            comm, tstart, tend);
               vftr_store_sync_message_info(recv, recvcount, recvtype, i, 0,
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
