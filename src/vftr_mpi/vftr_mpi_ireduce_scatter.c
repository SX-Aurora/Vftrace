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
#include "vftr_collective_requests.h"
#include "vftr_mpi_utils.h"
#include "vftr_mpi_buf_addr_const.h"

int vftr_MPI_Ireduce_scatter(const void *sendbuf, void *recvbuf, const int *recvcounts,
                             MPI_Datatype datatype, MPI_Op op, MPI_Comm comm,
                             MPI_Request *request) {

   // disable profiling based on the Pcontrol level
   if (vftr_no_mpi_logging()) {
      return PMPI_Ireduce_scatter(sendbuf, recvbuf, recvcounts, datatype, op, comm, request);
   } else {
      long long tstart = vftr_get_runtime_usec();
      int retVal = PMPI_Ireduce_scatter(sendbuf, recvbuf, recvcounts, datatype, op, comm, request);
  
      long long t2start = vftr_get_runtime_usec();
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
         // The Standard states in an advice to implementors, that MPI_Reduce_scatter behaves like
         // an MPI_Reduce followed by MPI_Scatterv. As root process we arbitrarily select 0 of each group
         int size;
         PMPI_Comm_size(comm, &size);
         int rank;
         PMPI_Comm_rank(comm, &rank);
         int remotesize;
         PMPI_Comm_remote_size(comm, &remotesize);

         // In both groups the sendbuffer size is the same.
         int count = 0;
         for (int i=0; i<size; i++) {
            count += recvcounts[i];
         }
         // determine global ranks for the local and remote root rank
         int remote_root_rank = vftr_remote2global_rank(comm, 0);
         int local_root_rank = vftr_local2global_rank(comm, 0);
         if (rank == 0) {
            // allocate memory for the temporary arrays
            // to register communication request
            int *tmpcount = (int*) malloc(sizeof(int)*(remotesize+1));
            MPI_Datatype *tmptype = (MPI_Datatype*) malloc(sizeof(MPI_Datatype)*(remotesize+1));
            int *peer_ranks = (int*) malloc(sizeof(int)*(remotesize+1));
            // messages received from remote ranks for reduction
            for (int i=0; i<remotesize; i++) {
               tmpcount[i] = count;
               tmptype[i] = datatype;
               // Register message info with MPI_COMM_WORLD as communicator
               // to prevent additional (and thus faulty rank translation)
               peer_ranks[i] = vftr_remote2global_rank(comm, i);
            }
            // scattered message received by itself 
            tmpcount[remotesize] = recvcounts[0];
            tmptype[remotesize] = datatype;
            // Register message info with MPI_COMM_WORLD as communicator
            // to prevent additional (and thus faulty rank translation)
            peer_ranks[remotesize] = local_root_rank;
            vftr_register_collective_request(recv, remotesize+1, tmpcount, tmptype, peer_ranks,
                                             MPI_COMM_WORLD, *request, tstart);
            if (size != remotesize) {
               // only reallocate if the two groups are of different size
               free(tmpcount);
               tmpcount = NULL;
               free(tmptype);
               tmptype = NULL;
               free(peer_ranks);
               peer_ranks = NULL;
   
               // allocate memory for the temporary arrays
               // to register communication request
               tmpcount = (int*) malloc(sizeof(int)*(size+1));
               tmptype = (MPI_Datatype*) malloc(sizeof(MPI_Datatype)*(size+1));
               peer_ranks = (int*) malloc(sizeof(int)*(size+1));
            }

            // process 0 scatters the reduction results to all members of its group
            for (int i=0; i<size; i++) {
               tmpcount[i] = recvcounts[i];
               tmptype[i] = datatype;
               // Register message info with MPI_COMM_WORLD as communicator
               // to prevent additional (and thus faulty rank translation)
               peer_ranks[i] = vftr_local2global_rank(comm, i);
            }
            // send the complete sendbuffer to process 0 of the remote group
            tmpcount[size] = count;
            tmptype[size] = datatype;
            // Register message info with MPI_COMM_WORLD as communicator
            // to prevent additional (and thus faulty rank translation)
            peer_ranks[size] = remote_root_rank;
            vftr_register_collective_request(send, size+1, tmpcount, tmptype, peer_ranks,
                                             MPI_COMM_WORLD, *request, tstart);
            free(tmpcount);
            tmpcount = NULL;
            free(tmptype);
            tmptype = NULL;
            free(peer_ranks);
            peer_ranks = NULL;
         } else {
            // send the complete sendbuffer to process 0 of the remote group
            // Register message info with MPI_COMM_WORLD as communicator
            // to prevent additional (and thus faulty rank translation)
            vftr_register_collective_request(send, 1, &count, &datatype,
                                             &remote_root_rank,
                                             MPI_COMM_WORLD, *request, tstart);
            // receive scattered reduction result from process 0 of own group
            // Register message info with MPI_COMM_WORLD as communicator
            // to prevent additional (and thus faulty rank translation)
            int recvcount = recvcounts[rank];
            vftr_register_collective_request(recv, 1, &recvcount, &datatype,
                                             &local_root_rank,
                                             MPI_COMM_WORLD, *request, tstart);
         }
      } else {
         int size;
         PMPI_Comm_size(comm, &size);
         int rank;
         PMPI_Comm_rank(comm, &rank);
         // if sendbuf is special address MPI_IN_PLACE
         if (vftr_is_C_MPI_IN_PLACE(sendbuf)) {
            if (size > 1) {
               // The sends and receives are not strictly true
               // as the number of peer processes with wich each
               // process communicates strongly depends on the unerlying reduction algorithm
               // There are multiple possibilities how to deal with this
               // 1. Reduce the data to one rank, and scatter it from there
               // 2. Every process functions as the root process of individual reduce operations
               //
               // The Standard states in an advice to implementors, that MPI_Reduce_scatter behaves like
               // an MPI_Reduce followed by MPI_Scatterv. As root process we arbitrarily select 0 of each group
               //
               // every rank needs to know the total size of the sendbuffers
               int count = 0;
               for (int i=0; i<size; i++) {
                  count += recvcounts[i];
               }
               if (rank == 0) {
                  // allocate memory for the temporary arrays
                  // to register communication request
                  int *tmpcount = (int*) malloc(sizeof(int)*(size-1));
                  MPI_Datatype *tmptype = (MPI_Datatype*) malloc(sizeof(MPI_Datatype)*(size-1));
                  int *peer_ranks = (int*) malloc(sizeof(int)*(size-1));
                  // messages to be received for reduction
                  for (int i=1; i<size; i++) {
                     tmpcount[i-1] = count;
                     tmptype[i-1] = datatype;
                     peer_ranks[i-1] = i;
                  }
                  vftr_register_collective_request(recv, size-1, tmpcount, tmptype, peer_ranks,
                                                   comm, *request, tstart);
                  // messages to be scatterd after reduction
                  for (int i=1; i<size; i++) {
                     // only need to adjust tmpcount to recvcount
                     // tmptype and peer_ranks are already correct
                     tmpcount[i-1] = recvcounts[i];
                  }
                  vftr_register_collective_request(send, size-1, tmpcount, tmptype, peer_ranks,
                                                   comm, *request, tstart);
                  free(tmpcount);
                  tmpcount = NULL;
                  free(tmptype);
                  tmptype = NULL;
                  free(peer_ranks);
                  peer_ranks = NULL;
               } else {
                  int root = 0;
                  // send to root process for reduction
                  vftr_register_collective_request(send, 1, &count, &datatype, &root,
                                                   comm, *request, tstart);
                  // receive result of redcution scattered to every process
                  int recvcount = recvcounts[rank];
                  vftr_register_collective_request(recv, 1, &recvcount, &datatype, &root,
                                                   comm, *request, tstart);
               }
            }
         } else {
            // The sends and receives are not strictly true
            // as the number of peer processes with wich each
            // process communicates strongly depends on the unerlying reduction algorithm
            // There are multiple possibilities how to deal with this
            // 1. Reduce the data to one rank, and scatter it from there
            // 2. Every process functions as the root process of individual reduce operations
            //
            // The Standard states in an advice to implementors, that MPI_Reduce_scatter behaves like
            // an MPI_Reduce followed by MPI_Scatterv. As root process we arbitrarily select 0 of each group
            //
            // every rank needs to know the total size of the sendbuffers
            int count = 0;
            for (int i=0; i<size; i++) {
               count += recvcounts[i];
            }
            if (rank == 0) {
               // allocate memory for the temporary arrays
               // to register communication request
               int *tmpcount = (int*) malloc(sizeof(int)*(size+1));
               MPI_Datatype *tmptype = (MPI_Datatype*) malloc(sizeof(MPI_Datatype)*(size+1));
               int *peer_ranks = (int*) malloc(sizeof(int)*(size+1));
               // messages to be received for reduction
               for (int i=0; i<size; i++) {
                  tmpcount[i] = count;
                  tmptype[i] = datatype;
                  peer_ranks[i] = i;
               }
               // scattered message received by itself
               tmpcount[size] = recvcounts[0];
               tmptype[size] = datatype;
               peer_ranks[size] = rank;
               vftr_register_collective_request(recv, size+1, tmpcount, tmptype, peer_ranks,
                                                comm, *request, tstart);
               // messages to be scatterd after reduction
               for (int i=0; i<size; i++) {
                  // only need to adjust tmpcount to recvcount
                  // tmptype and peer_ranks are already correct
                  tmpcount[i] = recvcounts[i];
               }
               // message send to itself process for reduction
               tmpcount[size] = count;
               vftr_register_collective_request(send, size+1, tmpcount, tmptype, peer_ranks,
                                                comm, *request, tstart);
               free(tmpcount);
               tmpcount = NULL;
               free(tmptype);
               tmptype = NULL;
               free(peer_ranks);
               peer_ranks = NULL;
            } else {
               int root = 0;
               // send to root process for reduction
               vftr_register_collective_request(send, 1, &count, &datatype, &root,
                                                comm, *request, tstart);
               // receive result of redcution scattered to every process
               int recvcount = recvcounts[rank];
               vftr_register_collective_request(recv, 1, &recvcount, &datatype, &root,
                                                comm, *request, tstart);
            }
         }
      }
      long long t2end = vftr_get_runtime_usec();

      vftr_mpi_overhead_usec += t2end - t2start;
  
      return retVal;
   }
}

#endif
