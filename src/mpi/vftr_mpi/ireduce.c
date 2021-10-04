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

#include <stdlib.h>

#include "vftr_timer.h"
#include "collective_requests.h"

int vftr_MPI_Ireduce(const void *sendbuf, void *recvbuf, int count,
                     MPI_Datatype datatype, MPI_Op op, int root, MPI_Comm comm,
                     MPI_Request *request) {
   long long tstart = vftr_get_runtime_usec();
   int retVal = PMPI_Ireduce(sendbuf, recvbuf, count, datatype, op, root, comm,
                             request);
   long long t2start = vftr_get_runtime_usec();
   // in intracommunicators the expected behaviour is to
   // bcast from root to all other processes in the communicator
   int rank;
   PMPI_Comm_rank(comm, &rank);
   if (rank == root) {
      int size;
      PMPI_Comm_size(comm, &size);
      // self communication
      vftr_register_collective_request(send, 1, &count, &datatype, &root,
                                       comm, *request, 0, NULL, tstart);
      // allocate memory for the temporary arrays
      // to register communication request
      int *tmpcount = (int*) malloc(sizeof(int)*size);
      MPI_Datatype *tmptype = (MPI_Datatype*) malloc(sizeof(MPI_Datatype)*size);
      int *peer_ranks = (int*) malloc(sizeof(int)*size);
      // messages to be received
      for (int i=0; i<size; i++) {
         tmpcount[i] = count;
         tmptype[i] = datatype;
         peer_ranks[i] = i;
      }
      vftr_register_collective_request(recv, size, tmpcount, tmptype, peer_ranks,
                                       comm, *request, 0, NULL, tstart);
      free(tmpcount);
      tmpcount = NULL;
      free(tmptype);
      tmptype = NULL;
      free(peer_ranks);
      peer_ranks = NULL;
   } else {
      vftr_register_collective_request(send, 1, &count, &datatype, &root,
                                       comm, *request, 0, NULL, tstart);
   }
   long long t2end = vftr_get_runtime_usec();

   vftr_mpi_overhead_usec += t2end - t2start;

   return retVal;
}

int vftr_MPI_Ireduce_inplace(const void *sendbuf, void *recvbuf, int count,
                             MPI_Datatype datatype, MPI_Op op, int root,
                             MPI_Comm comm,
                             MPI_Request *request) {
   long long tstart = vftr_get_runtime_usec();
   int retVal = PMPI_Ireduce(sendbuf, recvbuf, count, datatype, op, root, comm,
                             request);
   long long t2start = vftr_get_runtime_usec();
   // in intracommunicators the expected behaviour is to
   // bcast from root to all other processes in the communicator
   int rank;
   PMPI_Comm_rank(comm, &rank);
   if (rank == root) {
      int size;
      PMPI_Comm_size(comm, &size);
      if (size > 1) {
         // For the in-place option no self communication is executed

         // allocate memory for the temporary arrays
         // to register communication request
         int *tmpcount = (int*) malloc(sizeof(int)*(size-1));
         MPI_Datatype *tmptype = (MPI_Datatype*) malloc(sizeof(MPI_Datatype)*(size-1));
         int *peer_ranks = (int*) malloc(sizeof(int)*(size-1));
         // messages to be received
         int idx = 0;
         for (int i=0; i<rank; i++) {
            tmpcount[idx] = count;
            tmptype[idx] = datatype;
            peer_ranks[idx] = i;
            idx++;
         }
         for (int i=rank+1; i<size; i++) {
            tmpcount[idx] = count;
            tmptype[idx] = datatype;
            peer_ranks[idx] = i;
            idx++;
         }
         vftr_register_collective_request(recv, size-1, tmpcount, tmptype, peer_ranks,
                                          comm, *request, 0, NULL, tstart);
         free(tmpcount);
         tmpcount = NULL;
         free(tmptype);
         tmptype = NULL;
         free(peer_ranks);
         peer_ranks = NULL;
      }
   } else {
      vftr_register_collective_request(send, 1, &count, &datatype, &root,
                                       comm, *request, 0, NULL, tstart);
   }
   long long t2end = vftr_get_runtime_usec();

   vftr_mpi_overhead_usec += t2end - t2start;

   return retVal;
}

int vftr_MPI_Ireduce_intercom(const void *sendbuf, void *recvbuf, int count,
                              MPI_Datatype datatype, MPI_Op op, int root,
                              MPI_Comm comm, MPI_Request *request) {
   long long tstart = vftr_get_runtime_usec();
   int retVal = PMPI_Ireduce(sendbuf, recvbuf, count, datatype, op, root, comm,
                             request);
   long long t2start = vftr_get_runtime_usec();
   // in intercommunicators the behaviour is more complicated
   // There are two groups A and B
   // In group A the root process is located.
   if (root == MPI_ROOT) {
      // The root process get the special process wildcard MPI_ROOT
      // get the size of group B
      int size;
      PMPI_Comm_remote_size(comm, &size);
      int *tmpcount = (int*) malloc(sizeof(int)*size);
      MPI_Datatype *tmptype = (MPI_Datatype*) malloc(sizeof(MPI_Datatype)*size);
      int *peer_ranks = (int*) malloc(sizeof(int)*size);
      for (int i=0; i<size; i++) {
         tmpcount[i] = count;
         tmptype[i] = datatype;
         // translate the i-th rank in group B to the global rank
         peer_ranks[i] = vftr_remote2global_rank(comm, i);
      }
      // Register request with MPI_COMM_WORLD as communicator
      // to prevent additional (and thus faulty rank translation)
      vftr_register_collective_request(recv, size, tmpcount, tmptype, peer_ranks,
                                       MPI_COMM_WORLD, *request, 0, NULL, tstart);
      free(tmpcount);
      tmpcount = NULL;
      free(tmptype);
      tmptype = NULL;
      free(peer_ranks);
      peer_ranks = NULL;
   } else if (root == MPI_PROC_NULL) {
      // All other processes from group A pass wildcard MPI_PROC NULL
      // They do not participate in intercommunicator bcasts
      ;
   } else {
      // All other processes must be located in group B
      // root is the rank-id in group A Therefore no problems with 
      // rank translation should arise
      vftr_register_collective_request(send, 1, &count, &datatype, &root,
                                       comm, *request, 0, NULL, tstart);
   }
   long long t2end = vftr_get_runtime_usec();

   vftr_mpi_overhead_usec += t2end - t2start;

   return retVal;
}
