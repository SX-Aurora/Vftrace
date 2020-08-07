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
#include "vftr_sync_messages.h"
#include "vftr_mpi_pcontrol.h"
#include "vftr_mpi_buf_addr_const.h"

int vftr_MPI_Allreduce(const void *sendbuf, void *recvbuf, int count,
                       MPI_Datatype datatype, MPI_Op op, MPI_Comm comm) {

   // disable profiling based on the Pcontrol level
   if (vftrace_Pcontrol_level == 0) {
      return PMPI_Allreduce(sendbuf, recvbuf, count, datatype, op, comm);
   } else {
      long long tstart = vftr_get_runtime_usec();
      int retVal = PMPI_Allreduce(sendbuf, recvbuf, count, datatype, op, comm);
      long long tend = vftr_get_runtime_usec();
  
      // determine if inter or intra communicator
      int isintercom;
      PMPI_Comm_test_inter(comm, &isintercom);
      if (isintercom) {
         // Every process of group A performs the reduction within the group A
         // and stores the result on everyp process of group B and vice versa
         int size;
         PMPI_Comm_remote_size(comm, &size);
         for (int i=0; i<size; i++) {
            // translate the i-th rank in the remote group to the global rank
            int global_peer_rank = vftr_remote2global_rank(comm, i);
            // Store message info with MPI_COMM_WORLD as communicator
            // to prevent additional (and thus faulty rank translation)
            vftr_store_sync_message_info(send, count, datatype,
                                         global_peer_rank, -1, MPI_COMM_WORLD,
                                         tstart, tend);
            // The receive is not strictly true as every process receives only one 
            // data package, but due to the nature of a remote reduce
            // it is not possible to destinguish from whom.
            // There are three possibilities how to deal with this
            // 1. Don't register the receive at all
            // 2. Register the receive with count data from every remote process
            // 3. Register the receive with count/(remote size) data
            //    from every remote process
            // We selected number 2, because option 3 might not result
            // in an integer abmount of received data.
            vftr_store_sync_message_info(recv, count, datatype,
                                         global_peer_rank, -1, MPI_COMM_WORLD,
                                         tstart, tend);
         }
      } else {
         int size;
         PMPI_Comm_size(comm, &size);
         for (int i=0; i<size; i++) {
            vftr_store_sync_message_info(send, count, datatype, i, -1,
                                         comm, tstart, tend);
            vftr_store_sync_message_info(recv, count, datatype, i, -1,
                                         comm, tstart, tend);
         }
      }
  
      return retVal;
   }
}

#endif
