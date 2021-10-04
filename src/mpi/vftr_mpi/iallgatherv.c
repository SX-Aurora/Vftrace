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

int vftr_MPI_Iallgatherv(const void *sendbuf, int sendcount,
                         MPI_Datatype sendtype, void *recvbuf,
                         const int *recvcounts, const int *displs,
                         MPI_Datatype recvtype, MPI_Comm comm,
                         MPI_Request *request) {
   long long tstart = vftr_get_runtime_usec();
   int retVal = PMPI_Iallgatherv(sendbuf, sendcount, sendtype, recvbuf,
                                 recvcounts, displs, recvtype, comm,
                                 request);

   long long t2start = vftr_get_runtime_usec();
   int size;
   PMPI_Comm_size(comm, &size);
   // allocate memory for the temporary arrays
   // to register communication request
   int *tmpcount = (int*) malloc(sizeof(int)*size);
   MPI_Datatype *tmptype = (MPI_Datatype*) malloc(sizeof(MPI_Datatype)*size);
   int *peer_ranks = (int*) malloc(sizeof(int)*size);
   // messages to be send
   for (int i=0; i<size; i++) {
      tmpcount[i] = sendcount;
      tmptype[i] = sendtype;
      peer_ranks[i] = i;
   }
   // Store tmp-pointers for delayed deallocation
   int n_tmp_ptr = 2;
   void **tmp_ptrs = (void**) malloc(n_tmp_ptr*sizeof(void*));
   tmp_ptrs[0] = (void*) recvcounts;
   tmp_ptrs[1] = (void*) displs;
   vftr_register_collective_request(send, size, tmpcount, tmptype, peer_ranks,
                                    comm, *request, n_tmp_ptr, tmp_ptrs, tstart);
   // messages to be received
   for (int i=0; i<size; i++) {
      tmpcount[i] = recvcounts[i];
      tmptype[i] = recvtype;
      peer_ranks[i] = i;
   }
   vftr_register_collective_request(recv, size, tmpcount, tmptype, peer_ranks,
                                    comm, *request, 0, NULL, tstart);
   // cleanup temporary arrays
   free(tmpcount);
   tmpcount = NULL;
   free(tmptype);
   tmptype = NULL;
   free(peer_ranks);
   peer_ranks = NULL;
   long long t2end = vftr_get_runtime_usec();

   vftr_mpi_overhead_usec += t2end - t2start;

   return retVal;
}

int vftr_MPI_Iallgatherv_inplace(const void *sendbuf, int sendcount,
                                 MPI_Datatype sendtype, void *recvbuf,
                                 const int *recvcounts, const int *displs,
                                 MPI_Datatype recvtype, MPI_Comm comm,
                                 MPI_Request *request) {
   long long tstart = vftr_get_runtime_usec();
   int retVal = PMPI_Iallgatherv(sendbuf, sendcount, sendtype, recvbuf,
                                 recvcounts, displs, recvtype, comm,
                                 request);

   long long t2start = vftr_get_runtime_usec();
   int size;
   PMPI_Comm_size(comm, &size);
   if (size > 1) {
      int rank;
      PMPI_Comm_rank(comm, &rank);
      // For the in-place option no self communication is executed

      // allocate memory for the temporary arrays
      // to register communication request
      int *tmpcount = (int*) malloc(sizeof(int)*(size-1));
      MPI_Datatype *tmptype = (MPI_Datatype*) malloc(sizeof(MPI_Datatype)*(size-1));
      int *peer_ranks = (int*) malloc(sizeof(int)*(size-1));
      // messages to be send and received
      int idx = 0;
      for (int i=0; i<rank; i++) {
         tmpcount[idx] = recvcounts[rank];
         tmptype[idx] = recvtype;
         peer_ranks[idx] = i;
         idx++;
      }
      for (int i=rank+1; i<size; i++) {
         tmpcount[idx] = recvcounts[rank];
         tmptype[idx] = recvtype;
         peer_ranks[idx] = i;
         idx++;
      }
      // Store tmp-pointers for delayed deallocation
      int n_tmp_ptr = 2;
      void **tmp_ptrs = (void**) malloc(n_tmp_ptr*sizeof(void*));
      tmp_ptrs[0] = (void*) recvcounts;
      tmp_ptrs[1] = (void*) displs;
      vftr_register_collective_request(send, size-1, tmpcount, tmptype, peer_ranks,
                                       comm, *request, n_tmp_ptr, tmp_ptrs, tstart);
      idx = 0;
      for (int i=0; i<rank; i++) {
         tmpcount[idx] = recvcounts[i];
         tmptype[idx] = recvtype;
         peer_ranks[idx] = i;
         idx++;
      }
      for (int i=rank+1; i<size; i++) {
         tmpcount[idx] = recvcounts[i];
         tmptype[idx] = recvtype;
         peer_ranks[idx] = i;
         idx++;
      }
      vftr_register_collective_request(recv, size-1, tmpcount, tmptype, peer_ranks,
                                       comm, *request, 0, NULL, tstart);
      // cleanup temporary arrays
      free(tmpcount);
      tmpcount = NULL;
      free(tmptype);
      tmptype = NULL;
      free(peer_ranks);
      peer_ranks = NULL;
   }
   long long t2end = vftr_get_runtime_usec();

   vftr_mpi_overhead_usec += t2end - t2start;

   return retVal;
}

int vftr_MPI_Iallgatherv_intercom(const void *sendbuf, int sendcount,
                                  MPI_Datatype sendtype, void *recvbuf,
                                  const int *recvcounts, const int *displs,
                                  MPI_Datatype recvtype, MPI_Comm comm,
                                  MPI_Request *request) {
   long long tstart = vftr_get_runtime_usec();
   int retVal = PMPI_Iallgatherv(sendbuf, sendcount, sendtype, recvbuf,
                                 recvcounts, displs, recvtype, comm,
                                 request);

   long long t2start = vftr_get_runtime_usec();
   // Every process of group A sends sendcount data to and
   // receives recvcounts[i] data from the i-th process in group B
   int size;
   PMPI_Comm_remote_size(comm, &size);
   // allocate memory for the temporary arrays
   // to register communication request
   int *tmpcount = (int*) malloc(sizeof(int)*size);
   MPI_Datatype *tmptype = (MPI_Datatype*) malloc(sizeof(MPI_Datatype)*size);
   int *peer_ranks = (int*) malloc(sizeof(int)*size);
   // messages to be send
   for (int i=0; i<size; i++) {
      tmpcount[i] = sendcount;
      tmptype[i] = sendtype;
      // translate the i-th rank in the remote group to the global rank
      peer_ranks[i] = vftr_remote2global_rank(comm, i); 
   }
   // Store tmp-pointers for delayed deallocation
   int n_tmp_ptr = 2;
   void **tmp_ptrs = (void**) malloc(n_tmp_ptr*sizeof(void*));
   tmp_ptrs[0] = (void*) recvcounts;
   tmp_ptrs[1] = (void*) displs;
   // Register request with MPI_COMM_WORLD as communicator
   // to prevent additional (and thus faulty rank translation)
   vftr_register_collective_request(send, size, tmpcount, tmptype, peer_ranks,
                                    MPI_COMM_WORLD, *request,
                                    n_tmp_ptr, tmp_ptrs, tstart);
   // messages to be received
   for (int i=0; i<size; i++) {
      tmpcount[i] = recvcounts[i];
      tmptype[i] = recvtype;
      // translate the i-th rank in the remote group to the global rank
      peer_ranks[i] = vftr_remote2global_rank(comm, i); 
   }
   // Register request with MPI_COMM_WORLD as communicator
   // to prevent additional (and thus faulty rank translation)
   vftr_register_collective_request(recv, size, tmpcount, tmptype, peer_ranks,
                                    MPI_COMM_WORLD, *request, 0, NULL, tstart);
   // cleanup temporary arrays
   free(tmpcount);
   tmpcount = NULL;
   free(tmptype);
   tmptype = NULL;
   free(peer_ranks);
   peer_ranks = NULL;
   long long t2end = vftr_get_runtime_usec();

   vftr_mpi_overhead_usec += t2end - t2start;

   return retVal;
}
