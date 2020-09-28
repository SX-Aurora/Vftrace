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

int vftr_MPI_Ireduce_scatter_block(const void *sendbuf, void *recvbuf, int recvcount,
                                   MPI_Datatype datatype, MPI_Op op, MPI_Comm comm,
                                   MPI_Request *request) {

   // disable profiling based on the Pcontrol level
   if (vftrace_Pcontrol_level == 0) {
      return PMPI_Ireduce_scatter_block(sendbuf, recvbuf, recvcount, datatype, op, comm, request);
   } else {
      long long tstart = vftr_get_runtime_usec();
      int retVal = PMPI_Ireduce_scatter_block(sendbuf, recvbuf, recvcount, datatype, op, comm, request);
      long long tend = vftr_get_runtime_usec();
  
      // determine if inter or intra communicator
      int isintercom;
      PMPI_Comm_test_inter(comm, &isintercom);
      if (isintercom) {
         // Every process of group A performs the reduction within the group A
         // and stores the result on everyp process of group B and vice versa
         //
         // The sends and receives are not strictly true
         // as the number of peer processes with wich each
         // process communicates strongly depends on the unerlying reduction algorithm
         // There are multiple possibilities how to deal with this
         // 1. Reduce the data to one rank, and scatter it from there
         // 2. Every process functions as the root process of individual remote reduce operations
         //
         // To record the communication pattern of 1 a root process would need to be selected.
         // Therefore We selected number 2.
         //
         //
         // In both groups the sendbuffer size is the same.
         // Additionally it is required by the standard that
         // the send buffer size for MPI_Reduce_scatter_block with intracommunicators
         // is divisible by both group sizes.
         // Therefore the amount of data send to each remore process is
         // (local_group_size * recvcount) / remote_group_size
         //
         // translate the i-th rank in the remote group to the global rank
         int size;
         PMPI_Comm_size(comm, &size);
         int remotesize;
         PMPI_Comm_remote_size(comm, &remotesize);
         int *tmpcount = (int*) malloc(sizeof(int)*remotesize);
         MPI_Datatype *tmptype = (MPI_Datatype*) malloc(sizeof(MPI_Datatype)*remotesize);
         int *peer_ranks = (int*) malloc(sizeof(int)*remotesize);
         //messages to be send
         for (int i=0; i<remotesize; i++) {
            tmpcount[i] = (size * recvcount) / remotesize;
            tmptype[i] = datatype;
            // translate the i-th rank in the remote group to the global rank
            peer_ranks[i] = vftr_remote2global_rank(comm, i);
         }
         // Register request with MPI_COMM_WORLD as communicator
         // to prevent additional (and thus faulty rank translation)
         vftr_register_collective_request(send, remotesize, tmpcount, tmptype, peer_ranks,
                                          MPI_COMM_WORLD, *request, tstart);
         // messages to be received
         for (int i=0; i<remotesize; i++) {
            tmpcount[i] = recvcount;
            tmptype[i] = datatype;
            // translate the i-th rank in the remote group to the global rank
            peer_ranks[i] = vftr_remote2global_rank(comm, i);
         }
         // Register request with MPI_COMM_WORLD as communicator
         // to prevent additional (and thus faulty rank translation)
         vftr_register_collective_request(recv, size, tmpcount, tmptype, peer_ranks,
                                          MPI_COMM_WORLD, *request, tstart);
         // cleanup temporary arrays
         free(tmpcount);
         tmpcount = NULL;
         free(tmptype);
         tmptype = NULL;
         free(peer_ranks);
         peer_ranks = NULL;

      } else {
         int size;
         PMPI_Comm_size(comm, &size);
         // if sendbuf is special address MPI_IN_PLACE
         if (vftr_is_C_MPI_IN_PLACE(sendbuf)) {
            if (size > 1) {
               // For the in-place option no self communication is executed
               int rank;
               PMPI_Comm_rank(comm, &rank);

               // allocate memory for the temporary arrays
               // to register communication request
               int *tmpcount = (int*) malloc(sizeof(int)*(size-1));
               MPI_Datatype *tmptype = (MPI_Datatype*) malloc(sizeof(MPI_Datatype)*(size-1));
               int *peer_ranks = (int*) malloc(sizeof(int)*(size-1));
               // messages to be send
               int idx = 0;
               for (int i=0; i<rank; i++) {
                  tmpcount[idx] = recvcount;
                  tmptype[idx] = datatype;
                  peer_ranks[idx] = i;
                  idx++;
               }
               for (int i=rank+1; i<size; i++) {
                  tmpcount[idx] = recvcount;
                  tmptype[idx] = datatype;
                  peer_ranks[idx] = i;
                  idx++;
               }
               // The sends and receives are not strictly true
               // as the number of peer processes with wich each
               // process communicates strongly depends on the unerlying reduction algorithm
               // There are multiple possibilities how to deal with this
               // 1. Reduce the data to one rank, and scatter it from there
               // 2. Every process functions as the root process of individual reduce operations
               //
               // To record the communication pattern of 1 a root process would need to be selected.
               // Therefore We selected number 2.
               vftr_register_collective_request(send, size-1, tmpcount, tmptype, peer_ranks,
                                                comm, *request, tstart);
               vftr_register_collective_request(recv, size-1, tmpcount, tmptype, peer_ranks,
                                                comm, *request, tstart);
               // cleanup temporary arrays
               free(tmpcount);
               tmpcount = NULL;
               free(tmptype);
               tmptype = NULL;
               free(peer_ranks);
               peer_ranks = NULL;
            }
         } else {
            // allocate memory for the temporary arrays
            // to register communication request
            int *tmpcount = (int*) malloc(sizeof(int)*size);
            MPI_Datatype *tmptype = (MPI_Datatype*) malloc(sizeof(MPI_Datatype)*size);
            int *peer_ranks = (int*) malloc(sizeof(int)*size);
            // messages to be send
            for (int i=0; i<size; i++) {
               tmpcount[i] = recvcount;
               tmptype[i] = datatype;
               peer_ranks[i] = i;
            }
            // The sends and receives are not strictly true
            // as the number of peer processes with wich each
            // process communicates strongly depends on the unerlying reduction algorithm
            // There are multiple possibilities how to deal with this
            // 1. Reduce the data to one rank, and scatter it from there
            // 2. Every process functions as the root process of individual reduce operations
            //
            // To record the communication pattern of 1 a root process would need to be selected.
            // Therefore We selected number 2.
            vftr_register_collective_request(send, size, tmpcount, tmptype, peer_ranks,
                                             comm, *request, tstart);
            vftr_register_collective_request(recv, size, tmpcount, tmptype, peer_ranks,
                                             comm, *request, tstart);
            // cleanup temporary arrays
            free(tmpcount);
            tmpcount = NULL;
            free(tmptype);
            tmptype = NULL;
            free(peer_ranks);
            peer_ranks = NULL;
         }
      }
  
      return retVal;
   }
}

#endif
