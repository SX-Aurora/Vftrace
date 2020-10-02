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

#include <stdlib.h>

#include "vftr_timer.h"
#include "vftr_async_messages.h"
#include "vftr_mpi_pcontrol.h"
#include "vftr_mpi_buf_addr_const.h"

int vftr_MPI_Ialltoallw(const void *sendbuf, const int *sendcounts,
                        const int *sdispls, const MPI_Datatype *sendtypes,
                        void *recvbuf, const int *recvcounts, const int *rdispls,
                        const MPI_Datatype *recvtypes, MPI_Comm comm,
                        MPI_Request *request) {

   // disable profiling based on the Pcontrol level
   if (vftrace_Pcontrol_level == 0) {
      return PMPI_Ialltoallw(sendbuf, sendcounts, sdispls, sendtypes,
                             recvbuf, recvcounts, rdispls, recvtypes, comm,
                             request);
   } else {
      long long tstart = vftr_get_runtime_usec();
      int retVal = PMPI_Ialltoallw(sendbuf, sendcounts, sdispls, sendtypes,
                                   recvbuf, recvcounts, rdispls, recvtypes, comm,
                                   request);

      long long t2start = vftr_get_runtime_usec();
      // determine if inter or intra communicator
      int isintercom;
      PMPI_Comm_test_inter(comm, &isintercom);
      if (isintercom) {
         // Every process of group A sends sendcounts[i] sendtypes[i] to
         // and receives recvcounts[i] recvtypes[i] from
         // the i-th process in group B and vice versa.
         int size;
         PMPI_Comm_remote_size(comm, &size);
         // allocate memory for the temporary arrays
         // to register communication request
         int *tmpcount = (int*) malloc(sizeof(int)*size);
         MPI_Datatype *tmptype = (MPI_Datatype*) malloc(sizeof(MPI_Datatype)*size);
         int *peer_ranks = (int*) malloc(sizeof(int)*size);
         // messages to be send
         for (int i=0; i<size; i++) {
            tmpcount[i] = sendcounts[i];
            tmptype[i] = sendtypes[i];
            // translate the i-th rank in the remote group to the global rank
            peer_ranks[i] = vftr_remote2global_rank(comm, i);
         }
         // Register request with MPI_COMM_WORLD as communicator
         // to prevent additional (and thus faulty rank translation)
         vftr_register_collective_request(send, size, tmpcount, tmptype, peer_ranks,
                                          MPI_COMM_WORLD, *request, tstart);
         // messages to be received
         for (int i=0; i<size; i++) {
            tmpcount[i] = recvcounts[i];
            tmptype[i] = recvtypes[i];
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
         // sendcount and sendtype are ignored.
         // Use recvcount and recvtype for statistics
         if (vftr_is_C_MPI_IN_PLACE(sendbuf)) {
            if (size > 1) {
               int rank;
               PMPI_Comm_rank(comm, &rank);
               // For the in-place option no self communication is executed

               // allocate memory for the temporary arrays
               // to register communication request
               int *tmpcount = (int*) malloc(sizeof(int)*(size-1));
               MPI_Datatype *tmptype = (MPI_Datatype*) malloc(sizeof(MPI_Datatype)*(size-1));
               int *peer_ranks = (int*) malloc(sizeof(int)*(size-1));
               // messages to be send
               int idx = 0;
               for (int i=0; i<rank; i++) {
                  tmpcount[idx] = recvcounts[i];
                  tmptype[idx] = recvtypes[i];
                  peer_ranks[idx] = i;
                  idx++;
               }
               for (int i=rank+1; i<size; i++) {
                  tmpcount[idx] = recvcounts[i];
                  tmptype[idx] = recvtypes[i];
                  peer_ranks[idx] = i;
                  idx++;
               }
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
               tmpcount[i] = sendcounts[i];
               tmptype[i] = sendtypes[i];
               peer_ranks[i] = i;
            }
            vftr_register_collective_request(send, size, tmpcount, tmptype, peer_ranks,
                                             comm, *request, tstart);
            // messages to be received
            for (int i=0; i<size; i++) {
               tmpcount[i] = recvcounts[i];
               tmptype[i] = recvtypes[i];
               peer_ranks[i] = i;
            }
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
      long long t2end = vftr_get_runtime_usec();

      vftr_mpi_overhead_usec += t2end - t2start;

      return retVal;
   }
}

#endif
