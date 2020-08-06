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
#include "vftr_async_messages.h"
#include "vftr_mpi_pcontrol.h"
#include "vftr_mpi_buf_addr_const.h"

int vftr_MPI_Ialltoall(const void *sendbuf, int sendcount,
                       MPI_Datatype sendtype, void *recvbuf, int recvcount,
                       MPI_Datatype recvtype, MPI_Comm comm,
                       MPI_Request *request) {

   // disable profiling based on the Pcontrol level
   if (vftrace_Pcontrol_level == 0) {
      return PMPI_Ialltoall(sendbuf, sendcount, sendtype, recvbuf,
                           recvcount, recvtype, comm, request);
   } else {
      long long tstart = vftr_get_runtime_usec();
      int retVal = PMPI_Ialltoall(sendbuf, sendcount, sendtype, recvbuf,
                                 recvcount, recvtype, comm, request);
  
      // determine if inter or intra communicator
      int isintercom;
      PMPI_Comm_test_inter(comm, &isintercom);
      if (isintercom) {
         // Every process of group A sends sendcount sendtypes 
         // to and receives recvcount recvtypes from 
         // every process in group Band vice versa.
         int size;
         PMPI_Comm_remote_size(comm, &size);
         // allocate memory for the temporary arrays
         // to register communication request
         int *counts = (int*) malloc(sizeof(int)*size);
         MPI_Datatype *types = (MPI_Datatype*) malloc(sizeof(MPI_Datatype)*size);
         int *peer_ranks = (int*) malloc(sizeof(int)*size);
         // messages to be send
         for (int i=0; i<size; i++) {
            counts[i] = sendcount;
            types[i] = sendtype;
            // translate the i-th rank in the remote group to the global rank
            peer_ranks[i] = vftr_remote2global_rank(comm, i);
         }
         // Register request with MPI_COMM_WORLD as communicator
         // to prevent additional (and thus faulty rank translation)
         vftr_register_collective_request(send, size, counts, types, peer_ranks,
                                          MPI_COMM_WORLD, *request, tstart);
         // messages to be received
         for (int i=0; i<size; i++) {
            counts[i] = recvcount;
            types[i] = recvtype;
            // translate the i-th rank in the remote group to the global rank
            peer_ranks[i] = vftr_remote2global_rank(comm, i);
         }
         // Register request with MPI_COMM_WORLD as communicator
         // to prevent additional (and thus faulty rank translation)
         vftr_register_collective_request(recv, size, counts, types, peer_ranks,
                                          MPI_COMM_WORLD, *request, tstart);
         // cleanup temporary arrays
         free(counts);
         counts = NULL;
         free(types);
         types = NULL;
         free(peer_ranks);
         peer_ranks = NULL;
      } else {
         int size;
         PMPI_Comm_size(comm, &size);
         // if sendbuf is special address MPI_IN_PLACE
         // sendcount and sendtype are ignored.
         // Use recvcount and recvtype for statistics
         if (vftr_is_C_MPI_IN_PLACE(sendbuf)) {
            sendcount = recvcount;
            sendtype = recvtype;
         }
         // allocate memory for the temporary arrays
         // to register communication request
         int *counts = (int*) malloc(sizeof(int)*size);
         MPI_Datatype *types = (MPI_Datatype*) malloc(sizeof(MPI_Datatype)*size);
         int *peer_ranks = (int*) malloc(sizeof(int)*size);
         // messages to be send
         for (int i=0; i<size; i++) {
            counts[i] = sendcount;
            types[i] = sendtype;
            peer_ranks[i] = i;
         }
         vftr_register_collective_request(send, size, counts, types, peer_ranks,
                                          comm, *request, tstart);
         // messages to be received
         for (int i=0; i<size; i++) {
            counts[i] = recvcount;
            types[i] = recvtype;
            peer_ranks[i] = i;
         }
         vftr_register_collective_request(recv, size, counts, types, peer_ranks,
                                          comm, *request, tstart);
         // cleanup temporary arrays
         free(counts);
         counts = NULL;
         free(types);
         types = NULL;
         free(peer_ranks);
         peer_ranks = NULL;

      }
      return retVal;
   }
}

#endif
