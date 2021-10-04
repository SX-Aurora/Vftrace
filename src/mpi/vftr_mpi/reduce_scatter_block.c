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

#include <mpi.h>

#include "vftr_timer.h"
#include "sync_messages.h"

int vftr_MPI_Reduce_scatter_block(const void *sendbuf, void *recvbuf, int recvcount,
                                  MPI_Datatype datatype, MPI_Op op, MPI_Comm comm) {

   long long tstart = vftr_get_runtime_usec();
   int retVal = PMPI_Reduce_scatter_block(sendbuf, recvbuf, recvcount, datatype, op, comm);
   long long tend = vftr_get_runtime_usec();
  
   long long t2start = tend;
   int size;
   PMPI_Comm_size(comm, &size);
   int rank;
   PMPI_Comm_rank(comm, &rank);
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
   int count = size * recvcount;
   // send to root process for reduction
   vftr_store_sync_message_info(send, count, datatype, 0, -1,
                                comm, tstart, tend);
   if (rank == 0) {
      // First the Reduction on root:
      for (int i=0; i<size; i++) {
         // The root-rank receives count amount of data from each rank
         vftr_store_sync_message_info(recv, count, datatype, i, -1,
                                      comm, tstart, tend);
      }
      // Second the Scattering from root
      for (int i=0; i<size; i++) {
         vftr_store_sync_message_info(send, recvcount, datatype, i, -1,
                                      comm, tstart, tend);
      }
   }
   // receive result of redcution scattered to every process
   vftr_store_sync_message_info(recv, recvcount, datatype, 0, -1,
                                comm, tstart, tend);
   long long t2end = vftr_get_runtime_usec();

   vftr_mpi_overhead_usec += t2end - t2start;
  
   return retVal;
}

int vftr_MPI_Reduce_scatter_block_inplace(const void *sendbuf, void *recvbuf,
                                          int recvcount, MPI_Datatype datatype,
                                          MPI_Op op, MPI_Comm comm) {

   long long tstart = vftr_get_runtime_usec();
   int retVal = PMPI_Reduce_scatter_block(sendbuf, recvbuf, recvcount, datatype, op, comm);
   long long tend = vftr_get_runtime_usec();
  
   long long t2start = tend;
   int size;
   PMPI_Comm_size(comm, &size);
   int rank;
   PMPI_Comm_rank(comm, &rank);
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
      int count = size * recvcount;
      if (rank == 0) {
         // First the Reduction on root:
         for (int i=1; i<size; i++) {
            // The root-rank receives count amount of data from each rank
            vftr_store_sync_message_info(recv, count, datatype, i, -1,
                                         comm, tstart, tend);
         }
         // Second the Scattering from root
         for (int i=1; i<size; i++) {
            vftr_store_sync_message_info(send, recvcount, datatype, i, -1,
                                         comm, tstart, tend);
         }
      } else {
         // send to root process for reduction
         vftr_store_sync_message_info(send, count, datatype, 0, -1,
                                      comm, tstart, tend);
         // receive result of redcution scattered to every process
         vftr_store_sync_message_info(recv, recvcount, datatype, 0, -1,
                                      comm, tstart, tend);
      }
   }
   long long t2end = vftr_get_runtime_usec();

   vftr_mpi_overhead_usec += t2end - t2start;
  
   return retVal;
}

int vftr_MPI_Reduce_scatter_block_intercom(const void *sendbuf, void *recvbuf,
                                           int recvcount, MPI_Datatype datatype,
                                           MPI_Op op, MPI_Comm comm) {

   long long tstart = vftr_get_runtime_usec();
   int retVal = PMPI_Reduce_scatter_block(sendbuf, recvbuf, recvcount, datatype, op, comm);
   long long tend = vftr_get_runtime_usec();
  
   long long t2start = tend;
   // determine if inter or intra communicator
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
   int count = size*recvcount;
   // send the complete sendbuffer to process 0 of the remote group
   int global_peer_rank = vftr_remote2global_rank(comm, 0);
   // Store message info with MPI_COMM_WORLD as communicator
   // to prevent additional (and thus faulty rank translation)
   vftr_store_sync_message_info(send, count, datatype,
                                global_peer_rank, -1, MPI_COMM_WORLD,
                                tstart, tend);
   if (rank == 0) {
      // process 0 receives all sendbuffers from the remote group for reduction
      for (int i=0; i<remotesize; i++) {
         global_peer_rank = vftr_remote2global_rank(comm, i);
         // Store message info with MPI_COMM_WORLD as communicator
         // to prevent additional (and thus faulty rank translation)
         vftr_store_sync_message_info(recv, count, datatype,
                                      global_peer_rank, -1, MPI_COMM_WORLD,
                                      tstart, tend);
      }
      // process 0 scatters the reduction results to all members of its group
      for (int i=0; i<size; i++) {
         global_peer_rank = vftr_local2global_rank(comm, i);
         vftr_store_sync_message_info(send, recvcount, datatype,
                                      global_peer_rank, -1, MPI_COMM_WORLD,
                                      tstart, tend);
      }
   }

   // receive scattered reduction result from process 0 of own group
   global_peer_rank = vftr_local2global_rank(comm, 0);
   vftr_store_sync_message_info(recv, recvcount, datatype,
                                 global_peer_rank, -1, MPI_COMM_WORLD,
                                 tstart, tend);
   long long t2end = vftr_get_runtime_usec();

   vftr_mpi_overhead_usec += t2end - t2start;
  
   return retVal;
}
